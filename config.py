from pathlib import Path
from argparse import Namespace
import os
import torch 

def create_hparams():
    hparams = Namespace(batch_size = 4,
                        epochs = 50,
                        patience = 20,
                        lr = 1e-3,
                        sr = 16000,
                        n_mels = 96,
                        n_fft = 1024,
                        hop_len = 256,
                        audio_ms = 4080,
                        min_noise_ms = 1000,
                        num_channels = 1)
    
    audio_len_ = int(hparams.sr * hparams.audio_ms / 1000)
    frame_len_ = int(audio_len_ // hparams.hop_len + 1)
    hparams = Namespace(**vars(hparams),
                        audio_len = audio_len_,
                        frame_len = frame_len_)
    
    return hparams

device = 'cuda' if torch.cuda.is_available() else 'cpu'

MAIN_DIR = Path(__file__).parent
DATA_DIR = MAIN_DIR / "data"
WEIGHTS_DIR = MAIN_DIR / "weights"
RESULTS_DIR = MAIN_DIR / 'results'
PREDICTION_DIR = MAIN_DIR / 'predictions'
MIX_EX_DIR = MAIN_DIR / 'mixture_example'

# Make directories
if not os.path.exists(WEIGHTS_DIR):
    os.mkdir(WEIGHTS_DIR)

if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)
    
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
    
if not os.path.exists(PREDICTION_DIR):
    os.mkdir(PREDICTION_DIR)

if not os.path.exists(MIX_EX_DIR):
    os.mkdir(MIX_EX_DIR)
