import tensorflow as tf
import numpy as np
import librosa

import datasets

def spectrogram(noisy_speech: tf.Tensor, 
                clean_speech: tf.Tensor, 
                sr: int = 16000, 
                n_mels: int = 96, 
                n_fft: int = 1024, 
                hop_len: int = 259):
    """
    Computes the mel spectrogram of the noisy and the clean speech audio.
    It is used as a tf.py_function in datasets.prepare_enhancement_ds

    Parameters
    ----------
    noisy_speech : tf.Tensor
        The mixture of noise and speech.
    clean_speech : tf.Tensor
        The clean speech signal.
    sr : int, optional
        Sample Rate. The default is 16000.
    n_mels : int, optional
        Number of mels. The default is 96.
    n_fft : int, optional
        FFT length. The default is 1024.
    hop_len : int, optional
        Number of samples between successive frames.. The default is 259.

    Returns
    -------
    noisy_spec : tf.Tensor
        Spectrogram of the noisy speech audio.
    clean_spec : tf.Tensor
        Spectrogram of the clean speech audio.

    """
    
    noisy_spec = librosa.power_to_db(librosa.feature.melspectrogram(y=noisy_speech.numpy(), 
                                                                    sr = sr, 
                                                                    n_fft = n_fft, 
                                                                    hop_length = hop_len,
                                                                    n_mels = n_mels), ref=np.max, top_db=80.) / 80. + 1.
                                                            
    clean_spec = librosa.power_to_db(librosa.feature.melspectrogram(y=clean_speech.numpy(),
                                                                    sr = sr, 
                                                                    n_fft = n_fft, 
                                                                    hop_length = hop_len,
                                                                    n_mels = n_mels), ref=np.max, top_db=80.) / 80. + 1.
    
    noisy_spec = tf.expand_dims(tf.convert_to_tensor(noisy_spec), axis=-1)
    clean_spec = tf.expand_dims(tf.convert_to_tensor(clean_spec), axis=-1)

    return noisy_spec, clean_spec


def open_audio(noise_file: tf.Tensor, 
               speech_file: tf.Tensor, 
               mono: bool = True, 
               sr: int = 16000):
    """
    
    This is used as a tf.py_function to open the audio files
    Parameters
    ----------
    noise_file : tf.Tensor
        Noise audio path as tf.Tensor
    speech_file : tf.Tensor
        Speech audio path as tf.Tensor.
    mono : bool
        Specifies whether the audio is mono or stereo. The default is True.
    sr : int
        Sample rate. The default is 16000.

    Returns
    -------
    noisy_speech : np.ndarray
        Mixture of speech and noise.
    speech : np.ndarray
        Clean speech.

    """
    
    seed = np.random.randint(10000)
    speech, _ = librosa.load(speech_file.numpy(), mono=mono, sr=sr)
    noise, _ = librosa.load(noise_file.numpy(), mono=mono, sr=sr)
    
    noisy_speech, speech = datasets.generate_noisy_speech(speech, noise, sr=sr, seed=seed)
    
    # Zero mean, unitary variance normalization
    speech = (speech - speech.mean()) / speech.std()
    noisy_speech = (noisy_speech - noisy_speech.mean()) / noisy_speech.std()

    return noisy_speech, speech