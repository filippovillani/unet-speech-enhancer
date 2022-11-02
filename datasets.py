import numpy as np
import os
import librosa
import librosa.display
import pandas as pd
import tensorflow as tf


# TODO: Write the documenation for the functions

def build_noisy_speech_df(data_dir: str)->pd.DataFrame:
    # Loading the dataframe containing information about UrbanSound8K
    urban_metadata_path = os.path.join(data_dir, 'UrbanSound8K.csv')
    urban_df = pd.read_csv(urban_metadata_path)
    # Save the file paths in a new column of the dataframe
    urban_df['noise_path'] = data_dir + '/fold'  + urban_df['fold'].astype(str) + '/' + urban_df['slice_file_name'].astype(str)
    # Keep just the useful columns
    urban_df = urban_df[['noise_path', 'classID']]

    timit_metadata_path = os.path.join(data_dir, 'train_data.csv')
    timit_audio_dir = os.path.join(data_dir, 'data/')
    timit_df = pd.read_csv(timit_metadata_path)
    # timit_df = timit_df.rename(columns={"path_from_data_dir": "speech_path"})
    # I am just interested in the audio, I am dropping all the other files 
    timit_df = timit_df.loc[timit_df['is_audio']==True].loc[timit_df['is_converted_audio']==True]
    # Dropping all the columns but speech_path
    timit_df['speech_path'] = timit_audio_dir + timit_df['path_from_data_dir'].astype(str)
    timit_df = timit_df['speech_path']
    timit_df = timit_df.sample(frac=1.)
    timit_df = timit_df.reset_index(drop=True)
    
    num_classes = 10
    num_instances_per_class = int(len(timit_df) / num_classes)

    urban_df_reduced = urban_df[urban_df['classID']==0].sample(num_instances_per_class)

    for class_num in range(1, num_classes):
        class_instances = len(urban_df.loc[urban_df['classID']==class_num])
        if (class_instances < num_instances_per_class):
            urban_df_reduced = pd.concat([urban_df_reduced, urban_df[urban_df['classID']==class_num].sample(class_instances)])
            urban_df_reduced = pd.concat([urban_df_reduced, urban_df[urban_df['classID']==class_num].sample(num_instances_per_class - class_instances)])
        else:
            urban_df_reduced = pd.concat([urban_df_reduced, urban_df[urban_df['classID']==class_num].sample(num_instances_per_class)])

    # Shuffle the data and reset the indices
    urban_df_reduced = urban_df_reduced.sample(frac=1).reset_index(drop=True)

    noisy_speech_df =  urban_df_reduced.join(timit_df)

    noisy_speech_df = noisy_speech_df[['noise_path', 'speech_path', 'classID']]

    return noisy_speech_df


# These are the functions that preprocess the audio files. I'm going to use them to prepare the dataset.
def signal_power(signal):
    power = np.mean((np.abs(signal))**2)
    return power

def generate_noisy_speech(speech, noise, sr=16000, min_noise_ms=1000, audio_ms=4000, seed=None):
    np.random.seed(seed)
    
    SNR_premix = signal_power(speech) / signal_power(noise)
    SNR_in = np.random.uniform(-5, 5)
    SNR_in = np.power(10, SNR_in/10)
    noise_coefficient = np.sqrt(SNR_premix/SNR_in)
    
    noise_len = noise.shape[0]
    speech_len = speech.shape[0]
    
    audio_len = sr * audio_ms // 1000
    min_noise_len = sr * min_noise_ms // 1000
    
    # I want both noise and speech audio files to be the same length
    
    # if noise_len is below a certain threshold min_noise_len, then noise is duplicated.
    if noise_len <= min_noise_len:
        silence_len = np.random.randint(0, sr * 0.5) # I don't want more than 0.5s of silence
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


def open_audio(noise_file, speech_file, mono=True, sr=16000):
    seed = np.random.randint(10000)
    speech, _ = librosa.load(speech_file.numpy(), mono=mono, sr=sr)
    noise, _ = librosa.load(noise_file.numpy(), mono=mono, sr=sr)
    
    noisy_speech, speech = generate_noisy_speech(speech, noise, sr=sr, seed=seed)
    
    # Zero mean, unitary variance normalization
    speech = (speech - speech.mean()) / speech.std()
    noisy_speech = (noisy_speech - noisy_speech.mean()) / noisy_speech.std()

    return noisy_speech, speech

def spectrogram(noisy_speech, clean_speech, sr=16000, n_mels=96, n_fft=1024, hop_len=259):
    noisy_spec = librosa.power_to_db(librosa.feature.melspectrogram(noisy_speech.numpy(), 
                                                                    sr = sr, 
                                                                    n_fft = n_fft, 
                                                                    hop_length = hop_len,
                                                                    n_mels = n_mels), ref=np.max, top_db=80.) / 80. + 1.
                                                            
    clean_spec = librosa.power_to_db(librosa.feature.melspectrogram(clean_speech.numpy(),
                                                                    sr = sr, 
                                                                    n_fft = n_fft, 
                                                                    hop_length = hop_len,
                                                                    n_mels = n_mels), ref=np.max, top_db=80.) / 80. + 1.
    
    noisy_spec = tf.expand_dims(tf.convert_to_tensor(noisy_spec), axis=-1)
    clean_spec = tf.expand_dims(tf.convert_to_tensor(clean_spec), axis=-1)

    return noisy_spec, clean_spec


def prepare_enhancement_ds(ds, batch_size, train=False):
  if train:
    ds = ds.shuffle(len(ds))

  ds = ds.map(lambda noise, speech: tf.py_function(open_audio, 
                                                   [noise, speech], 
                                                   [tf.float32, tf.float32]), 
              num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(lambda noisy_speech, clean_speech: tf.py_function(spectrogram, 
                                                                [noisy_speech, clean_speech], 
                                                                [tf.float32, tf.float32]), 
              num_parallel_calls=tf.data.experimental.AUTOTUNE)

  ds = ds.cache()

  ds = ds.batch(batch_size)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  return ds

def build_ds_from_df(df, batch_size):
    df_len = len(df)
    train_len = int(0.8 * df_len)
    val_len = int(0.1 * df_len)

    train_df = df.iloc[:train_len].reset_index(drop=True)
    val_df = df.iloc[train_len:train_len + val_len].reset_index(drop=True)
    test_df = df.iloc[train_len + val_len:].reset_index(drop=True)


    train_ds = tf.data.Dataset.from_tensor_slices((train_df['noise_path'], 
                                                   train_df['speech_path']))
    train_ds = prepare_enhancement_ds(train_ds, batch_size=batch_size, train=True)

    val_ds = tf.data.Dataset.from_tensor_slices((val_df['noise_path'], 
                                                 val_df['speech_path'])) 
    val_ds = prepare_enhancement_ds(val_ds, batch_size=batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices((test_df['noise_path'], 
                                                  test_df['speech_path']))
    test_ds = prepare_enhancement_ds(test_ds, batch_size=batch_size)

    return train_ds, val_ds, test_ds