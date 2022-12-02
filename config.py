from pathlib import Path
from argparse import Namespace
import os

def create_hparams():
    hparams = Namespace(batch_size=16,
                        sr = 16000,
                        n_mels = 96,
                        n_fft = 1024,
                        hop_len = 259,
                        audio_ms = 4000,
                        min_noise_ms = 1000)
    return hparams


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
