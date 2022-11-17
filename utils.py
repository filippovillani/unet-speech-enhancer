import numpy as np
import tensorflow as tf
import librosa 

def signal_power(signal):
    """
    Computes the signal power

    Parameters
    ----------
    signal : np.ndarray

    Returns
    -------
    power : np.float
        The signal power.

    """
    
    power = np.mean((np.abs(signal))**2)
    return power


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

def waveform_from_spectrogram(spectrogram: np.ndarray, 
                              sr: int = 16000, 
                              n_fft: int = 1024, 
                              hop_len: int = 259, 
                              n_iter:int = 512):
    """
    Computes the waveform of an audio given its spectrogram through Griffin-Lim Algorithm


    Parameters
    ----------
    spectrogram : np.ndarray
        Spectrogram to be converted to waveform.
    sr : int, optional
        Sample rate. The default is 16000.
    n_fft : int, optional
        FFT length. The default is 1024.
    hop_len : int, optional
        Number of samples between successive frames.. The default is 259.
    n_iter : int, optional
        Number of iterations for the algorithm. The default is 512.

    Returns
    -------
    inverse_spectrogram : np.ndarray
        The waveform obtained by Griffin-Lim Algorithm.

    """

    
    spectrogram = (spectrogram - 1.) * 80.
    inverse_spectrogram = librosa.feature.inverse.mel_to_audio(librosa.db_to_power(spectrogram), 
                                                               sr = sr, 
                                                               n_fft = n_fft, 
                                                               hop_length = hop_len, 
                                                               n_iter = n_iter)
    return inverse_spectrogram

