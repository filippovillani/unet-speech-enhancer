from argparse import Namespace
from typing import Tuple

import pandas as pd
import torch
from librosa.filters import mel as melfb
from torch.utils.data import DataLoader, Dataset, random_split

import config
from utils.audioutils import (normalize_db_spectr, open_audio, signal_power,
                              standardization, to_db)


def build_dataloaders(hparams: Namespace,
                      data_dir: str)->Tuple[DataLoader, DataLoader, DataLoader]: 
    
    ds = NoisySpeechDataset(hparams, data_dir)
    train_ds, val_ds, test_ds = random_split(ds, [0.7, 0.15, 0.15])
    train_dl = DataLoader(train_ds, 
                          hparams.batch_size, 
                          shuffle=True, 
                          drop_last=True)
    val_dl = DataLoader(val_ds, 
                        hparams.batch_size, 
                        shuffle=False,
                        drop_last=True)
    test_dl = DataLoader(test_ds, 
                         hparams.batch_size, 
                         shuffle=False,
                         drop_last=True)
    return train_dl, val_dl, test_dl


class NoisySpeechDataset(Dataset):
    def __init__(self, 
                 hparams: Namespace,
                 data_dir: str):
        
        super().__init__()
        self.data_dir = data_dir
        self.hprms = hparams
        self.urban_df = self._build_noise_df()
        self.timit_df = self._build_speech_df()
        self.melfb = torch.as_tensor(melfb(sr = self.hprms.sr, 
                                           n_fft = self.hprms.n_fft, 
                                           n_mels = self.hprms.n_mels))
        
        
    def _build_noise_df(self):
        
        urban_metadata_path = self.data_dir / 'UrbanSound8K.csv'
        urban_df = pd.read_csv(urban_metadata_path)
        urban_df['noise_path'] = self.data_dir / ('fold'  + urban_df['fold'].astype(str)) / urban_df['slice_file_name'].astype(str)
        urban_df['noise_path'] = urban_df['noise_path'].astype(str)
        urban_df = urban_df[['noise_path', 'classID']]
        
        return urban_df


    def _build_speech_df(self):
        
        timit_metadata_path = self.data_dir / 'train_data.csv'
        timit_audio_dir = self.data_dir / 'data'
        timit_df = pd.read_csv(timit_metadata_path)
        timit_df = timit_df.loc[timit_df['is_audio']==True].loc[timit_df['is_converted_audio']==True]
        # Dropping all the columns but speech_path
        timit_df['speech_path'] = timit_audio_dir / timit_df['path_from_data_dir'].astype(str)
        timit_df = timit_df['speech_path'].astype(str)
        timit_df = timit_df.sample(frac=1.)
        timit_df = timit_df.reset_index(drop=True)
        
        return timit_df


    def _generate_noisy_speech(self,
                               speech: torch.Tensor,
                               noise: torch.Tensor, 
                               sr: int = 16000,
                               audio_ms = 2040,
                               min_noise_ms = 1000,
                               max_snr_db = 5,
                               min_snr_db = -5)->torch.Tensor:
        """
        Blends environmental noise from UrbanSound8K with utterances from TIMIT.
        The noise power for each sample is a uniform random variable with values in [-5, 5] dB.
        

        Parameters
        ----------
        speech : torch.Tensor
            clean speech signal
        noise : torch.Tensor
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
        noisy_speech: torch.Tensor
            Mixture of speech and noise.
        speech : torch.Tensor
            Clean speech.

        """        
                        
        noise_len = noise.shape[0]
        speech_len = speech.shape[0]
        
        audio_len = sr * audio_ms // 1000
        min_noise_len = sr * min_noise_ms // 1000
        
        # I want both noise and speech audio paths to be the same length
        
        # if noise_len is below a certain threshold min_noise_len, then noise is duplicated.
        if noise_len <= min_noise_len:
            silence_len = torch.randint(0, int(sr * 0.5), (1,)) # I don't want more than 0.5s of silence
            silence = torch.zeros(silence_len)
            noise = torch.cat([noise, silence, noise])  
            noise_len = noise.shape[0]
        
        # if the current length of noise is above max_len then noise is truncated.
        if noise_len > audio_len:
            start_position = torch.randint(0, noise_len - audio_len, (1,))
            end_position = start_position + audio_len
            noise = noise[start_position:end_position]  
        
        # if the current length of noise is below max_len then noise is padded.
        elif noise_len < audio_len:
            pad_begin_len = torch.randint(0, audio_len - noise_len, (1,))
            pad_end_len = audio_len - noise_len - pad_begin_len
            
            pad_begin = torch.zeros(pad_begin_len)
            pad_end = torch.zeros(pad_end_len)
            
            noise = torch.cat([pad_begin, noise, pad_end])
        
        # Speech truncation
        if speech_len > audio_len:
            start_position = torch.randint(0, speech_len - audio_len, (1,))
            end_position = start_position + audio_len
            speech = speech[start_position:end_position]
        # Speech pad
        elif speech_len < audio_len:
            pad_begin_len = torch.randint(0, audio_len - speech_len, (1,))
            pad_end_len = audio_len - speech_len - pad_begin_len
            
            pad_begin = torch.zeros(pad_begin_len)
            pad_end = torch.zeros(pad_end_len)
        
            speech = torch.cat([pad_begin, speech, pad_end])  
        
        
        SNR_premix = signal_power(speech) / signal_power(noise)
        SNR_in = (max_snr_db - min_snr_db) * torch.rand((1)) + min_snr_db
        SNR_in = torch.pow(10, SNR_in/10)
        noise_coefficient = torch.sqrt(SNR_premix/SNR_in)
        noisy_speech = (speech + noise_coefficient * noise)
                
        return noisy_speech, speech  
    
    
    def __getitem__(self, 
                    idx)->dict:
        
        noise_idx = torch.randint(0, len(self.urban_df), (1,)).item()
        
        speech_path = self.timit_df[idx]
        noise_path = self.urban_df["noise_path"][noise_idx]
        
        data = {"noisy": [],
                "speech": [],
                "noisy_phasegram": [],
                "clean_speech_wav": []}
        
        speech = open_audio(speech_path, sr=self.hprms.sr)
        noise = open_audio(noise_path, sr=self.hprms.sr)
        
        noisy_speech, speech = self._generate_noisy_speech(speech, 
                                                           noise, 
                                                           sr=self.hprms.sr,
                                                           audio_ms = self.hprms.audio_ms,
                                                           min_noise_ms = self.hprms.min_noise_ms,
                                                           max_snr_db = self.hprms.max_snr_db,
                                                           min_snr_db = self.hprms.min_snr_db)
        
        # Zero mean, unitary variance normalization
        speech = standardization(speech)
        noisy_speech = standardization(noisy_speech)
        
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
        
        data["speech"] = normalize_db_spectr(to_db(torch.matmul(self.melfb, 
                                                        (torch.abs(speech_stft)**2).float()), 
                                            power_spectr=True)).unsqueeze(0)
        
        data["noisy"] = normalize_db_spectr(to_db(torch.matmul(self.melfb, 
                                                               (torch.abs(noisy_speech_stft)**2).float()),
                                                  power_spectr=True)).unsqueeze(0)
        
        data["noisy_phasegram"] = torch.angle(noisy_speech_stft)
        data["clean_speech_wav"] = speech
            
            
        return data
    
    
    def __len__(self):
        return len(self.timit_df)
    
    
if __name__ == "__main__":

    data_dir = config.DATA_DIR
    hparams = config.create_hparams()
    
    train_dl, _, _ = build_dataloaders(hparams, data_dir)
    for el in train_dl:
        print(type(el))
    
    
    
    

