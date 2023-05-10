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

def create_awgn(signal, max_snr_db = 12, min_snr_db = -6):

    snr_db = (max_snr_db - min_snr_db) * torch.rand((1)) + min_snr_db
    snr = torch.pow(10, snr_db/10).to(signal.device)

    signal_power = torch.mean(torch.abs(signal) ** 2)
    
    noise_power = signal_power / (snr + 1e-12)
    noise = torch.sqrt(noise_power) * torch.randn_like(signal)
    
    return noise