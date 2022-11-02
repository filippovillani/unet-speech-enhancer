import librosa 

def waveform_from_spectrogram(spectrogram, sr=16000, n_fft=1024, hop_len=259, n_iter=512):
    spectrogram = (spectrogram - 1.) * 80.
    inverse_spectrogram = librosa.feature.inverse.mel_to_audio(librosa.db_to_power(spectrogram), 
                                                               sr = sr, 
                                                               n_fft = n_fft, 
                                                               hop_length = hop_len, 
                                                               n_iter = n_iter)

    return inverse_spectrogram