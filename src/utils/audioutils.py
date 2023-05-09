import torch
import numpy as np
import librosa
import soundfile as sf


def open_audio(audio_path, sr):
    
    audio, _ = librosa.load(audio_path, sr=sr)
    audio = torch.as_tensor(audio)
    audio = standardization(audio)
    return audio

def save_audio(x_wav, x_wav_path, sr = 16000):
    
    if isinstance(x_wav, torch.Tensor):
        x_wav = x_wav.cpu().detach().numpy()
    x_wav = min_max_normalization(x_wav.squeeze()) * 2 - 1
    sf.write(x_wav_path, x_wav, sr)


def to_db(spectrogram, power_spectr = False, min_db = -80):
    
    scale = 10 if power_spectr else 20
    spec_max = torch.max(spectrogram)
    spec_db = torch.clamp(scale * torch.log10(spectrogram / spec_max + 1e-12), min=min_db, max=0)
    return spec_db


def to_linear(spectrogram_db):
    
    spec_lin = torch.pow(10, spectrogram_db / 20)
    return spec_lin


def normalize_db_spectr(spectrogram):
    return (spectrogram / 80) + 1


def denormalize_db_spectr(spectrogram):
    return (spectrogram - 1) * 80

def min_max_normalization(x_wav):
    
    if isinstance(x_wav, torch.Tensor):
        x_wav = (x_wav - torch.min(x_wav)) / (torch.max(x_wav) - torch.min(x_wav))
    if isinstance(x_wav, np.ndarray):
        x_wav = (x_wav - np.min(x_wav)) / (np.max(x_wav) - np.min(x_wav))
    return x_wav

def standardization(x_wav):
    return (x_wav - x_wav.mean()) / (x_wav.std() + 1e-12)

def signal_power(signal):
    
    power = torch.mean((torch.abs(signal))**2)
    return power


def melspectrogram(audio: np.ndarray,
                   sr: int = 16000,
                   n_mels: int = 96, 
                   n_fft: int = 1024, 
                   hop_len: int = 256)->np.ndarray:
    
    melspectrogram = librosa.power_to_db(librosa.feature.melspectrogram(y=audio,
                                                                        sr = sr, 
                                                                        n_fft = n_fft, 
                                                                        hop_length = hop_len,
                                                                        n_mels = n_mels), ref=np.max, top_db=80.) / 80. + 1.
    
    
    return melspectrogram

def inverse_spectrogram(melspectrogram: np.ndarray, 
                        sr: int = 16000, 
                        n_fft: int = 1024, 
                        hop_len: int = 256, 
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
        Number of samples between successive frames.. The default is 256.
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
