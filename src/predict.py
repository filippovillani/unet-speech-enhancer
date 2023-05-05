import numpy as np
import soundfile as sf
import tensorflow as tf
import librosa 

import config
from networks.UNet.models import UNet
from utils.audioutils import melspectrogram, inverse_spectrogram
from utils.plots import plot_spectrograms

def predict(args):
    # TODO: to pytorch
    weights_path = config.WEIGHTS_DIR / args.weights_dir / args.weights_dir
    audio_path = config.MIX_EX_DIR / args.audio_path
    output_path = config.PREDICTION_DIR / audio_path.name.replace(".wav", f"_prediction_{args.weights_dir}.wav")
    fig_path = config.PREDICTION_DIR / audio_path.name.replace(".wav", f"_{args.weights_dir}.png")
    # Compute input spectrogram, make the prediction and save waveform
    audio, _ = librosa.load(str(audio_path), sr=16000) 
    spectr = tf.expand_dims(melspectrogram(audio), axis=0)
    # Initialize the model and load weights
    model = UNet().double().to(config.device)
    # TODO: load torch weights
    model.load_weights(weights_path)

    # spectr = tf.expand_dims(spectrogram(audio), axis=0)
    enhanced_speech_spectr = model(spectr, training=False)
    enhanced_speech_spectr = tf.squeeze(enhanced_speech_spectr)
    # Reconstruct the time-domain signal
    enhanced_speech = inverse_spectrogram(enhanced_speech_spectr)
    enhanced_speech = enhanced_speech / np.max(enhanced_speech)
    
    # Computed spectrogram of inverse_spectrogram of enhanced_speech for plot
    reconstructed_spec = melspectrogram(enhanced_speech)
    
    sf.write(output_path, enhanced_speech, samplerate=16000)
    plot_spectrograms(tf.squeeze(spectr), enhanced_speech_spectr, tf.squeeze(reconstructed_spec), fig_path)
    