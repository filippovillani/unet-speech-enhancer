import soundfile as sf

import config
from model import UNet
from utils import open_audio, spectrogram, inverse_spectrogram

def predict(args):
    weights_path = config.WEIGHTS_DIR / args.weights_path
    audio_path = config.MIX_EX_DIR / args.audio_path
    model = UNet()
    model.build = True
    model.load_weights(weights_path)
    
    audio = open_audio(audio_path)
    # compute spectrogram
    spectr = spectrogram(audio)
    enhanced_speech_spectr = model(spectr, training=False)
    enhanced_speech = inverse_spectrogram(enhanced_speech_spectr)
    
    output_path = config.PREDICTION_DIR / audio_path.name.replace(".wav", "_prediction.wav")
    sf.write(output_path, enhanced_speech)
