import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_pe(hidden_size, max_len=100):  
    pe = torch.zeros(max_len, hidden_size)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_size))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return pe


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResBottleneck(nn.Module):
    def __init__(self, hidden_size, norm=True) -> None:
        super().__init__()
        self.norm = norm

        self.down = nn.Linear(hidden_size, hidden_size // 4)
        self.activation = nn.LeakyReLU()
        self.up = nn.Linear(hidden_size // 4, hidden_size)
        
        if self.norm:
            self.ln = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        y = self.up(self.activation(self.down(x))) + x
        if self.norm:
            return self.ln(y)
        else:
            return y

class ImageAdapter(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()

        self.down = nn.Linear(in_dim, in_dim // 4)
        self.activation = nn.LeakyReLU()
        self.up = nn.Linear(in_dim // 4, in_dim)
        
        self.out_linear = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        y = self.up(self.activation(self.down(x))) + x
        return self.out_linear(y)


class WrappedTransformerEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        dropout: int,
        n_layers: int = None,
        n_encoder_layers: int = None,
        **kwargs
    ):
        super().__init__()
        if n_encoder_layers is not None:
            n_layers = n_encoder_layers
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=n_heads,
                dim_feedforward=hidden_size*4,
                dropout=dropout,
                activation=F.gelu,
                batch_first=True,
                norm_first=True
            ),
            num_layers=n_layers
        )
        self.ln = nn.LayerNorm(hidden_size)
    
    def forward(self, embeddings):
        return self.ln(self.encoder(embeddings))


class WrappedTransformerDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        dropout: int,
        n_layers: int = None,
        n_decoder_layers: int = None,
        **kwargs
    ):
        super().__init__()
        if n_decoder_layers is not None:
            n_layers = n_decoder_layers
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=n_heads,
                dim_feedforward=hidden_size*4,
                dropout=dropout,
                activation=F.gelu,
                batch_first=True,
                norm_first=True
            ),
            num_layers=n_layers
        )
        self.ln = nn.LayerNorm(hidden_size)
    
    def forward(self, tgt, memory):
        return self.ln(self.decoder(tgt, memory))
