import numpy as np
import soundfile as sf
import tensorflow as tf

import config
from model import UNet, unet
from utils import open_audio, spectrogram, inverse_spectrogram

def predict(args):
    weights_path = config.WEIGHTS_DIR / args.weights_dir / args.weights_dir
    audio_path = config.MIX_EX_DIR / args.audio_path
    output_path = config.PREDICTION_DIR / audio_path.name.replace(".wav", f"_prediction_{args.weights_dir}.wav")

    # Compute input spectrogram, make the prediction and save waveform
    audio = open_audio(audio_path)
    spectr = tf.expand_dims(spectrogram(audio), axis=0)
    # Initialize the model and load weights
    model = UNet.build_model(input_size=(96, 248, 1))
    model.load_weights(weights_path)
    

    # spectr = tf.expand_dims(spectrogram(audio), axis=0)
    enhanced_speech_spectr = model(spectr, training=False)
    
    enhanced_speech_spectr = tf.squeeze(enhanced_speech_spectr)
    enhanced_speech = inverse_spectrogram(enhanced_speech_spectr)
    enhanced_speech = enhanced_speech / np.max(enhanced_speech)
    
    sf.write(output_path, enhanced_speech, samplerate=16000)