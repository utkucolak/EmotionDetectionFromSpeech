import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from utils.create_mel_spectogram import *
from utils.constants import EMOTION_TO_IDX, EMOTION_MAP_CREMAD
import random

class RAVDESSDataset(Dataset):
    def __init__(self, csv_path, max_len=300):
        self.df = pd.read_csv(csv_path)
        self.max_len = max_len
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        filepath = row["filepath"]
        label = EMOTION_TO_IDX[row["emotion"]]

        mel_spec = create_mel_spectogram(filepath)

        if mel_spec.shape[1] < self.max_len:
            pad_width = self.max_len - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0,0), (0, pad_width)), mode="constant")
        else:
            mel_spec = mel_spec[:, :self.max_len]
        
        mel_spec_tensor = torch.tensor(mel_spec, dtype=torch.float32)
        mel_spec_tensor = mel_spec_tensor.transpose(0, 1) # (time_steps, n_mels)

        return mel_spec_tensor, label



class CREMADDataset(Dataset):
    def __init__(self, csv_path, max_len=300, sr=22050, use_augmentation=False):
        self.df = pd.read_csv(csv_path)
        self.max_len = max_len
        self.sr = sr
        self.use_augmentation = use_augmentation

    def __len__(self):
        return len(self.df)

    def augment_audio(self, y, sr=22050):
    # Volume change
        gain = random.uniform(0.8, 1.2)
        y = y * gain
    
        # Pitch shift
        n_steps = random.choice([-2, -1, 0, 1, 2])
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    
        # Add noise
        noise_amp = 0.005 * np.random.uniform() * np.amax(y)
        y = y + noise_amp * np.random.normal(size=y.shape)
    
        return y

    def __getitem__(self, index):
        row = self.df.iloc[index]
        filepath = row["filepath"]
        label_str = row["emotion"]
        label = EMOTION_MAP_CREMAD[label_str.upper()[:3]]  # 'disgust' → 'DIS'

        y, _ = librosa.load(filepath, sr=self.sr)

        if self.use_augmentation:
            y = self.augment_audio(y)

        mel = create_mel_spectogram(filepath)  # normalized (n_mels, time_steps)

        # Pad or truncate time axis
        if mel.shape[1] < self.max_len:
            pad_width = self.max_len - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant')
        elif mel.shape[1] > self.max_len:
            mel = mel[:, :self.max_len]

        mel_tensor = torch.tensor(mel).transpose(0, 1).float()  # (T, n_mels)

        return mel_tensor, label
    