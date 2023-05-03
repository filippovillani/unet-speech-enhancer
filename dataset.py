import numpy as np
import librosa
import librosa.display
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from argparse import Namespace
from typing import Tuple

from utils import signal_power, standardization, to_db, normalize_db_spectr


def build_dataloaders(data_dir: str, 
                      hparams: Namespace)->Tuple[Dataset, Dataset, Dataset]: 
    
    ds = NoisySpeechDataset(data_dir, hparams)
    train_ds, val_ds, test_ds = random_split(ds, [0.7, 0.15, 0.15], generator=torch.Generator().manual_seed(42))
    train_dl = DataLoader(train_ds, 
                          hparams.batch_size, 
                          shuffle=True, 
                          pin_memory=True)
    val_dl = DataLoader(val_ds, 
                        hparams.batch_size, 
                        shuffle=False,
                        pin_memory=True)
    test_dl = DataLoader(test_ds, 
                         hparams.batch_size, 
                         shuffle=False,
                         pin_memory=True)
    return train_dl, val_dl, test_dl


class NoisySpeechDataset(Dataset):
    def __init__(self, 
                 data_dir: str,
                 hparams: Namespace):
        
        super().__init__()
        self.data_dir = data_dir
        self.hprms = hparams
        self.df = self._build_noisy_speech_dfs()
        self.melfb = torch.as_tensor(librosa.filters.mel(sr = self.hprms.sr, 
                                                         n_fft = self.hprms.n_fft, 
                                                         n_mels = self.hprms.n_mels)).to(self.hprms.device)
    
    
    def _build_noisy_speech_dfs(self):
        """
        Builds a pd.DataFrame that contains the paths to the noise and speech
        audio paths, along with the noise classID.

        Returns
        -------
        noisy_speech_df : pd.DataFrame
            'noise_path': str, 'speech_path': str, 'classID': int

        """
        
        seed = 42
        
        # Loading the dataframe containing information about UrbanSound8K
        urban_metadata_path = self.data_dir / 'UrbanSound8K.csv'
        urban_df = pd.read_csv(urban_metadata_path)
        # Save the path paths in a new column of the dataframe
        urban_df['noise_path'] = self.data_dir / ('fold'  + urban_df['fold'].astype(str)) / urban_df['slice_file_name'].astype(str)
        # Keep just the useful columns
        urban_df['noise_path'] = urban_df['noise_path'].astype(str)
        urban_df = urban_df[['noise_path', 'classID']]

        timit_metadata_path = self.data_dir / 'train_data.csv'
        timit_audio_dir = self.data_dir / 'data'
        timit_df = pd.read_csv(timit_metadata_path)
        timit_df = timit_df.loc[timit_df['is_audio']==True].loc[timit_df['is_converted_audio']==True]
        # Dropping all the columns but speech_path
        timit_df['speech_path'] = timit_audio_dir / timit_df['path_from_data_dir'].astype(str)
        timit_df = timit_df['speech_path'].astype(str)
        timit_df = timit_df.sample(frac=1., random_state=seed)
        timit_df = timit_df.reset_index(drop=True)
        
        num_classes = 10
        num_instances_per_class = int(len(timit_df) / num_classes)

        urban_df_reduced = urban_df[urban_df['classID']==0].sample(num_instances_per_class, random_state=seed)

        for class_num in range(1, num_classes):
            class_instances = len(urban_df.loc[urban_df['classID']==class_num])
            if (class_instances < num_instances_per_class):
                urban_df_reduced = pd.concat([urban_df_reduced, urban_df[urban_df['classID']==class_num].sample(class_instances, random_state=seed)])
                urban_df_reduced = pd.concat([urban_df_reduced, urban_df[urban_df['classID']==class_num].sample(num_instances_per_class - class_instances, random_state=seed)])
            else:
                urban_df_reduced = pd.concat([urban_df_reduced, urban_df[urban_df['classID']==class_num].sample(num_instances_per_class, random_state=seed)])

        # Shuffle the data and reset the indices
        urban_df_reduced = urban_df_reduced.sample(frac=1, random_state=seed).reset_index(drop=True)

        noisy_speech_df =  urban_df_reduced.join(timit_df)

        noisy_speech_df = noisy_speech_df[['noise_path', 'speech_path', 'classID']]

        return noisy_speech_df

    def _generate_noisy_speech(self,
                               speech: np.ndarray,
                               noise: np.ndarray, 
                               seed = None)->np.ndarray:
        """
        Blends environmental noise from UrbanSound8K with utterances from TIMIT.
        The noise power for each sample is a uniform random variable with values in [-5, 5] dB.
        

        Parameters
        ----------
        speech : np.ndarray
            clean speech signal
        noise : np.ndarray
            noise signal
        sr : int
            Sample rate. The default is 16000Hz.
        min_noise_ms : int
            Minimum duration of noise audio. If the audio duration is below this threshold then it will be
            duplicated in the mixed audio. The default is 1000ms.
        audio_ms : int
            Mixed audio duration. The default is 4000ms.

        Returns
        -------
        noisy_speech: np.ndarray
            Mixture of speech and noise.
        speech : np.ndarray
            Clean speech.

        """
        
        np.random.seed(seed)
        
        SNR_premix = signal_power(speech) / signal_power(noise)
        SNR_in = np.random.uniform(-5, 5)
        SNR_in = np.power(10, SNR_in/10)
        noise_coefficient = np.sqrt(SNR_premix/SNR_in)
        
        noise_len = noise.shape[0]
        speech_len = speech.shape[0]
        
        audio_len = self.hprms.sr * self.hprms.audio_ms // 1000
        min_noise_len = self.hprms.sr * self.hprms.min_noise_ms // 1000
        
        # I want both noise and speech audio paths to be the same length
        
        # if noise_len is below a certain threshold min_noise_len, then noise is duplicated.
        if noise_len <= min_noise_len:
            silence_len = np.random.randint(0, self.hprms.sr * 0.5) # I don't want more than 0.5s of silence
            silence = np.zeros(silence_len)
            noise = np.concatenate((noise, silence, noise))   
            noise_len = noise.shape[0]
        
        # if the current length of noise is above max_len then noise is truncated.
        if noise_len > audio_len:
            start_position = np.random.randint(0, noise_len - audio_len)
            end_position = start_position + audio_len
            noise = noise[start_position:end_position]  
        
        # if the current length of noise is below max_len then noise is padded.
        elif noise_len < audio_len:
            pad_begin_len = np.random.randint(0, audio_len - noise_len)
            pad_end_len = audio_len - noise_len - pad_begin_len
            
            pad_begin = np.zeros(pad_begin_len)
            pad_end = np.zeros(pad_end_len)
            
            noise = np.concatenate((pad_begin, noise, pad_end))
        
        # Speech truncation
        if speech_len > audio_len:
            start_position = np.random.randint(0, speech_len - audio_len)
            end_position = start_position + audio_len
            speech = speech[start_position:end_position]
        # Speech pad
        elif speech_len < audio_len:
            pad_begin_len = np.random.randint(0, audio_len - speech_len)
            pad_end_len = audio_len - speech_len - pad_begin_len
            
            pad_begin = np.zeros(pad_begin_len)
            pad_end = np.zeros(pad_end_len)
        
            speech = np.concatenate((pad_begin, speech, pad_end))  
                    
        return (speech + noise_coefficient * noise), speech        
  
    def __getitem__(self, 
                    idx)->dict:
        
        seed = np.random.randint(10000)
        speech_path = self.df["speech_path"][idx]
        noise_path = self.df["noise_path"][idx]
        
        data = {"noisy": [],
                "speech": []}
        
        speech, _ = librosa.load(speech_path, sr=self.hprms.sr)
        noise, _ = librosa.load(noise_path, sr=self.hprms.sr)
        
        noisy_speech, speech = self._generate_noisy_speech(speech, noise, seed=seed)
        
        # Zero mean, unitary variance normalization
        speech = torch.as_tensor(standardization(speech))
        noisy_speech = torch.as_tensor(standardization(noisy_speech))
        
        speech_stft = torch.stft(speech,
                                 n_fft=self.hprms.n_fft,
                                 hop_length=self.hprms.hop_len,
                                 window = torch.hann_window(self.hprms.n_fft),
                                 return_complex=True)
        
        noisy_speech_stft = torch.stft(noisy_speech,
                                       n_fft=self.hprms.n_fft,
                                       hop_length=self.hprms.hop_len,
                                       window = torch.hann_window(self.hprms.n_fft),
                                       return_complex=True)
        
        # Spectrograms and convert to torch.tensor
        data["speech"] = normalize_db_spectr(to_db(torch.matmul(self.melfb, 
                                                                (torch.abs(speech_stft)**2).float()), 
                                                   power_spectr=True)).unsqueeze(0)
        
        data["noisy"] = normalize_db_spectr(to_db(torch.matmul(self.melfb, 
                                                               (torch.abs(noisy_speech_stft)**2).float()),
                                                  power_spectr=True)).unsqueeze(0)
            
        return data
    
    def __len__(self):
        return len(self.df)

if __name__ == "__main__":
    import config
    train_dl, val_dl, test_dl = build_dataloaders(config.DATA_DIR, config.create_hparams())
    for el in train_dl:
        print(el["speech"].shape)
        break