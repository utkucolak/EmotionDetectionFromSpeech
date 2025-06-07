import torch
import torch.nn as nn
import numpy as np
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, time_steps: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.time_steps = time_steps
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (time_steps, d_model)

        pe_matrix = torch.zeros(time_steps, d_model)

        position = torch.arange(0, time_steps, dtype=torch.float).unsqueeze(1) #vector size (time_steps, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sin to even positions, cosine to the odd positions.
        pe_matrix[:, 0::2] = torch.sin(position * div_term)
        pe_matrix[:, 1::2] = torch.cos(position * div_term)

        pe_matrix = pe_matrix.unsqueeze(0) # (1, time_steps, d_model)

        self.register_buffer('pe_matrix', pe_matrix)

    def forward(self, x):
        x = x + (self.pe_matrix[:, :x.shape[1], :]).requires_grad_(False) #this particular tensor is not learned.
        return self.dropout(x)