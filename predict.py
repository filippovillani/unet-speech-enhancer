import numpy as np
import soundfile as sf
import tensorflow as tf

import config
from model import UNet, unet
from utils import open_audio, spectrogram, inverse_spectrogram

def predict(args):
    weights_path = config.WEIGHTS_DIR / args.weights_dir /args.weights_dir
    audio_path = config.MIX_EX_DIR / args.audio_path
    output_path = config.PREDICTION_DIR / audio_path.name.replace(".wav", f"_prediction_{args.weights_dir}.wav")

    # Initialize the model and load weights
    model = unet()
    model.build = True
    model.load_weights(weights_path)
    
    audio = open_audio(audio_path)

    # Compute input spectrogram, make the prediction and save waveform
    spectr = tf.expand_dims(spectrogram(audio), axis=0)
    enhanced_speech_spectr = model(spectr, training=False)
    enhanced_speech_spectr = tf.squeeze(enhanced_speech_spectr)
    enhanced_speech = inverse_spectrogram(enhanced_speech_spectr)
    enhanced_speech = enhanced_speech / np.max(enhanced_speech)
    
    sf.write(output_path, enhanced_speech, samplerate=16000)

