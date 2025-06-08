import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from utils.create_mel_spectogram import *
from utils.constants import EMOTION_TO_IDX

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