import matplotlib.pyplot as plt
import librosa
import librosa.display

def plot_spectrograms(noisy_spec,
                      enhanced_spec,
                      reconstructed_spec,
                      fig_path):

    noisy_spec = noisy_spec.numpy()
    enhanced_spec = enhanced_spec.numpy()
    reconstructed_spec = reconstructed_spec.numpy()

    plt.figure()
    plt.subplot(3,1,1)
    librosa.display.specshow(noisy_spec, sr=16000)
    plt.subplot(3,1,2)
    librosa.display.specshow(enhanced_spec, sr=16000)
    plt.subplot(3,1,3)
    librosa.display.specshow(reconstructed_spec, sr=16000)
    plt.savefig(fig_path)