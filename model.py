import torch
import torch.nn as nn
import numpy as np
import math
# The formulas of the blocks in this script comes from the paper "Attention Is All You Need", Vaswani A. et. al, 2023
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
    
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplier
        self.bias = nn.Parameter(torch.zeros(1)) # Adder

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    # FFN(x) = max(0, xW1 + b1)W2 + b
    def __init__(self, d_model : int, d_ff : int, dropout : float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) #W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (Batch, time_steps, d_model) ---> (Batch, time_steps, d_ff) ---> (Batch, time_steps, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    # time_steps = time length
    # d_model = size of the embedding vector
    # h = number of heads
    # d_k = d_v = d_model // h
    def __init__(self, d_model : int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask = None, dropout: nn.Dropout = None):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # produces a tensor shape: (batch, h, time_steps, time_steps)
        
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        attention_scores = attention_scores.softmax(dim = -1)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores


    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch, time_steps, d_model) ---> (batch, time_steps, h, d_k) ---> (batch, h, time_steps, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k) # (batch, h, time_steps, d_k) ---> (batch, time_steps, h, d_k) ---> (batch, time_steps, d_model)

        # (batch, time_steps, d_model) ---> (batch, time_steps, d_model)
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    #skip connection while feeding the encoder.
    def __init__(self, dropout : float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)]) # 2 residual connections exist in the encoder according to the paper.

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) # pass x as query, key and value
        x = self.residual_connections[1](x, self.feed_forward_block)

        return x

# encoder may consists of N encoder blocks

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class ClassifierLayer(nn.Module):

    def __init__(self, d_model: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.mean(dim=1) # (batch, d_model)

        return self.classifier(x) # (batch, num_classes)

class TransformerClassifier(nn.Module):
    
    def __init__(self, encoder: Encoder, src_pos: PositionalEncoding, classifier_layer: ClassifierLayer, d_model: int, n_mels: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.src_pos = src_pos
        self.classifier_layer = classifier_layer
        self.input_proj = nn.Linear(n_mels, d_model)

    def forward(self, x, mask=None):
        x = self.input_proj(x)
        x = self.src_pos(x)                     # Apply positional encoding 
        x = self.encoder(x, mask)               # Feed the encoder
        x = self.classifier_layer(x)            # Final classification layer
        return x
    
def build_transformer(
    time_steps: int,            # input sequence length (e.g., mel frames)
    d_model: int = 128,         # size of feature embedding (keep manageable)
    N: int = 4,                 # number of encoder blocks
    h: int = 4,                 # number of attention heads
    dropout: float = 0.3,
    d_ff: int = 512,            # feedforward inner dim
    num_classes: int = 8,        # number of output emotion classes
    n_mels: int = 64
) -> TransformerClassifier:
    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, time_steps, dropout)

    encoder_blocks = []

    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))

    classifier_layer = ClassifierLayer(d_model, num_classes)

    transformer = TransformerClassifier(encoder, src_pos, classifier_layer, d_model, n_mels)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer