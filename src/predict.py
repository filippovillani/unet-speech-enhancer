import argparse

import torch
from librosa.filters import mel

import config
from networks.PInvDAE.models import PInvDAE
from networks.UNet.models import UNet
from utils.audioutils import (denormalize_db_spectr, normalize_db_spectr,
                              open_audio, save_audio, standardization, to_db,
                              to_linear)
from utils.utils import load_config


def predict(args):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    noisy_speech_path = config.MIX_EX_DIR / args.audio
    enh_speech_path = config.MIX_EX_DIR / args.audio.replace('.wav', '_enh.wav')
    enh_weights_path = config.MODELS_DIR / args.enhancer / 'weights' / 'best_weights'
    enh_config_path = config.MODELS_DIR / args.enhancer / 'config.json'
    pinv_weights_path = config.MODELS_DIR / args.pinv / 'weights' / 'best_weights'
    pinv_config_path = config.MODELS_DIR / args.pinv / 'config.json'
    
    print('Loading models...')
    enh_hparams = load_config(enh_config_path)
    enh_model = UNet(enh_hparams).to(device)
    enh_model.load_state_dict(torch.load(enh_weights_path))
    
    pinv_hparams = load_config(pinv_config_path)
    pinv_model = PInvDAE(pinv_hparams).to(device)
    pinv_model.load_state_dict(torch.load(pinv_weights_path))
    
    print('Processing audio...')
    noisy_wav = open_audio(noisy_speech_path, pinv_hparams.sr)
    noisy_wav = standardization(noisy_wav)
    noisy_stft = torch.stft(noisy_wav,
                            n_fft=pinv_hparams.n_fft,
                            hop_length=pinv_hparams.hop_len,
                            window = torch.hann_window(pinv_hparams.n_fft),
                            return_complex=True).to(device)
    melfb = torch.as_tensor(mel(sr = pinv_hparams.sr, 
                                n_fft = pinv_hparams.n_fft, 
                                n_mels = pinv_hparams.n_mels)).to(device)
    
    noisy_melspec = normalize_db_spectr(to_db(torch.matmul(melfb, 
                                                           (torch.abs(noisy_stft)**2).float()), 
                                              power_spectr=True)).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        
        enh_melspec = enh_model(noisy_melspec)
        enh_stftspec = pinv_model(enh_melspec).squeeze(1)
        enh_stftspec = to_linear(denormalize_db_spectr(enh_stftspec))
        enh_stft = enh_stftspec * torch.exp(1j * torch.angle(noisy_stft))
        enh_speech_wav = torch.istft(enh_stft,
                                     n_fft = pinv_hparams.n_fft,
                                     window = torch.hann_window(pinv_hparams.n_fft).to(enh_stft.device)).squeeze()

    save_audio(enh_speech_wav, enh_speech_path)
    print('Done! You can find your enhanced speech at')
    print(enh_speech_path)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--enhancer', 
                        type=str, 
                        default='enhancer80_00') 
    
    parser.add_argument('--pinv', 
                        type=str, 
                        default='pinvdae80_00') 
    
    parser.add_argument('--audio', 
                        type=str,
                        help="audio.wav in mixture_example/", 
                        default='noisy0.wav') 
    
    args = parser.parse_args()
    
    predict(args)