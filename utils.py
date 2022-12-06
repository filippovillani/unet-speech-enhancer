import numpy as np
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


def melspectrogram(audio: np.ndarray,
                   sr: int = 16000,
                   n_mels: int = 96, 
                   n_fft: int = 1024, 
                   hop_len: int = 259)->np.ndarray:
    
    melspectrogram = librosa.power_to_db(librosa.feature.melspectrogram(y=audio,
                                                                        sr = sr, 
                                                                        n_fft = n_fft, 
                                                                        hop_length = hop_len,
                                                                        n_mels = n_mels), ref=np.max, top_db=80.) / 80. + 1.
    
    
    return melspectrogram

def inverse_spectrogram(melspectrogram: np.ndarray, 
                        sr: int = 16000, 
                        n_fft: int = 1024, 
                        hop_len: int = 259, 
                        n_iter:int = 512)->np.ndarray:
    """
    Computes the waveform of an audio given its melspectrogram through Griffin-Lim Algorithm


    Parameters
    ----------
    melspectrogram : np.ndarray
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

    
    melspectrogram = (melspectrogram - 1.) * 80.
    inverse_spectrogram = librosa.feature.inverse.mel_to_audio(librosa.db_to_power(melspectrogram), 
                                                               sr = sr, 
                                                               n_fft = n_fft, 
                                                               hop_length = hop_len, 
                                                               n_iter = n_iter)
    return inverse_spectrogram

