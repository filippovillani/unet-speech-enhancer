import soundfile as sf
import tensorflow as tf

import config
from model import UNet, unet
from utils import open_audio, spectrogram, inverse_spectrogram

def predict(args):
    weights_path = config.WEIGHTS_DIR / args.weights_path
    audio_path = config.MIX_EX_DIR / args.audio_path
    output_path = config.PREDICTION_DIR / audio_path.name.replace(".wav", "_prediction.wav")

    model = unet()
    model.build = True
    model.load_weights(weights_path)
    
    audio = open_audio(audio_path)
    # compute spectrogram
    spectr = tf.expand_dims(spectrogram(audio), axis=0)
    enhanced_speech_spectr = model(spectr, training=False)
    enhanced_speech_spectr = tf.squeeze(enhanced_speech_spectr)
    enhanced_speech = inverse_spectrogram(enhanced_speech_spectr)
    
    sf.write(output_path, enhanced_speech, samplerate=16000)

