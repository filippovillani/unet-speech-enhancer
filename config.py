from pathlib import Path
import os

MAIN_DIR = Path(__file__).parent
DATA_DIR = MAIN_DIR / "data"
WEIGHTS_DIR = MAIN_DIR / "checkpoint"
TRAINING_STATE_DIR = MAIN_DIR / 'training_states'
PREDICTION_DIR = MAIN_DIR / 'predictions'
MIX_EX_DIR = MAIN_DIR / 'mixture_example'

# Make directories
if not os.path.exists(WEIGHTS_DIR):
    os.mkdir(WEIGHTS_DIR)

if not os.path.exists(TRAINING_STATE_DIR):
    os.mkdir(TRAINING_STATE_DIR)
    
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
    
if not os.path.exists(PREDICTION_DIR):
    os.mkdir(PREDICTION_DIR)

if not os.path.exists(MIX_EX_DIR):
    os.mkdir(MIX_EX_DIR)

# TODO
def set_seed():
    pass