import librosa 
import numpy as np

def create_mel_spectogram(path : str):
    scale, sr = librosa.load(path)
    mel_spectogram = librosa.feature.melspectrogram(y=scale, sr=sr, n_fft=2048, hop_length=512, n_mels=64)
    mel_spec_db = librosa.power_to_db(mel_spectogram, ref=np.max)
    mel_spec_db = (mel_spec_db - np.mean(mel_spec_db)) / (np.std(mel_spec_db) + 1e-9)
    return mel_spec_db

def create_mel_spectogram_for_inference(waveform: np.ndarray, sr: int = 22050, n_mels: int = 64, max_len: int = 300) -> np.ndarray:

    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=2048, hop_length=512, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # (n_mels, time_steps)
    mel_spec_db = (mel_spec_db - np.mean(mel_spec_db)) / (np.std(mel_spec_db) + 1e-9)

    return mel_spec_db  # (n_mels, time_steps)