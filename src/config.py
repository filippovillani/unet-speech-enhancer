import os
import random
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch


def create_hparams():   
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    training_hparams = Namespace(batch_size = 1,
                                 lr = 1e-4,
                                 weights_decay = 1e-4,
                                 epochs = 30,
                                 patience = 10,
                                 lr_patience = 3,
                                 loss = "l1", # can be one of ["l1", "complexmse", "mse", "frobenius"]
                                 max_snr_db = 5,
                                 min_snr_db = -5) 
                                 
    model_hparams = Namespace(first_unet_channel_units = 32,
                              unet_kernel_size = (3,3),
                              drop_rate = 0)
    
    audio_hparams = Namespace(sr = 16000,
                              n_mels = 80,
                              n_fft = 1024,
                              n_channels = 1,
                              hop_len = 256,
                              audio_ms = 2040, #4080
                              audio_thresh = 0.1,
                              min_noise_ms = 1000)
    # Other useful audio parameters
    audio_len_ = int(audio_hparams.sr * audio_hparams.audio_ms // 1000)
    n_frames_ = int(audio_len_ // audio_hparams.hop_len + 1)
    n_stft_ = int(audio_hparams.n_fft//2 + 1)
    
    hparams = Namespace(**vars(training_hparams),
                        **vars(model_hparams),
                        **vars(audio_hparams),
                        audio_len = audio_len_,
                        n_frames = n_frames_,
                        n_stft = n_stft_,
                        device = device,
                        num_workers = 2)
    
    return hparams

def set_seeds(seed = 42):
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# For reproducibility
SEED = 42
set_seeds(SEED)

# Directories
MAIN_DIR = Path(__file__).parent.parent
sys.path.append(str(MAIN_DIR))
sys.path.append(str(MAIN_DIR/'src'))

DATA_DIR = MAIN_DIR / "data"
WEIGHTS_DIR = MAIN_DIR / "weights"
RESULTS_DIR = MAIN_DIR / 'results'
PREDICTION_DIR = MAIN_DIR / 'predictions'
MIX_EX_DIR = MAIN_DIR / 'mixture_example'

_dirs = [WEIGHTS_DIR, DATA_DIR, RESULTS_DIR, PREDICTION_DIR, MIX_EX_DIR]

for dir in _dirs:
    if not os.path.exists(dir):
        os.mkdir(dir)