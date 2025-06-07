import librosa 
import numpy as np

def create_mel_spectogram(path : str):
    scale, sr = librosa.load(path)
    mel_spectogram = librosa.feature.melspectrogram(y=scale, sr=sr, n_fft=2048, hop_length=512, n_mels=10)
    mel_spec_db = librosa.power_to_db(mel_spectogram, ref=np.max)
    return mel_spec_db