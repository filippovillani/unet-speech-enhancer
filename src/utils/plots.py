import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path

from utils.utils import load_json

def plot_spectrograms(noisy_spec,
                      clean_spec,
                      enhanced_spec,
                      fig_path = None):

    noisy_spec = noisy_spec.numpy()
    clean_spec = clean_spec.numpy()
    enhanced_spec = enhanced_spec.numpy()

    plt.figure()
    plt.subplot(3,1,1)
    librosa.display.specshow(noisy_spec, sr=16000)
    plt.subplot(3,1,2)
    librosa.display.specshow(clean_spec, sr=16000)
    plt.subplot(3,1,3)
    librosa.display.specshow(enhanced_spec, sr=16000)
    if fig_path is not None:
        plt.savefig(fig_path)   
        
        
def plot_train_hist(training_state_path: Path):
 
    training_state = load_json(training_state_path)

    for metric in training_state["train_hist"].keys():
        save_path = training_state_path.parent / (metric + '.png')
        plt.figure(figsize=(10, 8))
        plt.plot(range(1, 1+training_state["epochs"]), training_state["train_hist"][metric], label='train')
        plt.plot(range(1, 1+training_state["epochs"]), training_state["val_hist"][metric], label='validation')
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.legend()
        plt.grid()
        
        plt.savefig(save_path)
        plt.close()