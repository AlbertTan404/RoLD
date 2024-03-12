import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from RoLD.models.common import get_pe


class Action2Embedding(nn.Module):
    def __init__(
        self,
        action_dim: int,
        hidden_size: int,
        horizon: int,
        **kwargs
    ):
        super().__init__()
        self.action_emb = nn.Linear(action_dim, hidden_size)
        self.register_buffer('pe', get_pe(hidden_size=hidden_size, max_len=2*horizon))

    def forward(self, actions):
        embeddings = self.action_emb(actions)
        embeddings = embeddings + self.pe[:, :actions.size(1), : ]
        return embeddings

class Embedding2Latent(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        dropout: int,
        n_layers: int,
        **kwargs
    ):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=n_heads,
                dim_feedforward=hidden_size*4,
                dropout=dropout,
                activation=F.gelu,
                batch_first=True
            ),
            num_layers=n_layers
        )
        self.head = nn.Linear(hidden_size, hidden_size*2)
    
    def forward(self, embeddings):
        latent = self.encoder(embeddings)
        return self.head(latent)


class ConditionedEmbedding2Latent(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        dropout: int,
        n_layers: int,
        **kwargs
    ):
        super().__init__()
        self.encoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=n_heads,
                dim_feedforward=hidden_size*4,
                dropout=dropout,
                activation=F.gelu,
                batch_first=True
            ),
            num_layers=n_layers
        )
        self.head = nn.Linear(hidden_size, hidden_size*2)
    
    def forward(self, embeddings, condition):
        latent = self.encoder(embeddings, memory=condition)
        return self.head(latent)


class Action2Latent(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.action2emb = Action2Embedding(**kwargs)
        self.emb2latent = Embedding2Latent(**kwargs)

    def forward(self, actions):
        embeddings = self.action2emb(actions)
        return self.emb2latent(embeddings)


class Decoder(nn.Module):
    def __init__(
        self,
        action_dim: int,
        hidden_size: int,
        n_heads: int,
        dropout: int,
        n_layers: int,
        horizon: int,
        **kwargs
    ) -> None:
        super().__init__()

        self.register_buffer('pe', get_pe(hidden_size=hidden_size, max_len=2*horizon))
        self.pos_dropout = nn.Dropout(p=dropout)

        self.decoder = nn.TransformerEncoder(
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=n_heads,
                dim_feedforward=4*hidden_size,
                dropout=dropout,
                activation=F.gelu,
                batch_first=True
            ),
            num_layers=n_layers
        )
        self.action_head = nn.Linear(hidden_size, action_dim, bias=False)

    def forward(self, latents):
        latents = self.pos_dropout(latents + self.pe[:, :latents.size(1), : ])
        latents = self.decoder(latents)
        return self.action_head(latents)


class ConditionedDecoder(nn.Module):
    def __init__(
        self,
        action_dim: int,
        hidden_size: int,
        n_heads: int,
        dropout: int,
        n_layers: int,
        horizon: int,
        **kwargs
    ) -> None:
        super().__init__()

        self.register_buffer('pe', get_pe(hidden_size=hidden_size, max_len=2*horizon))

        self.pos_dropout = nn.Dropout(p=dropout)

        self.decoder = nn.TransformerDecoder(
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=n_heads,
                dim_feedforward=4*hidden_size,
                dropout=dropout,
                activation=F.gelu,
                batch_first=True
            ),
            num_layers=n_layers
        )
        self.action_head = nn.Linear(hidden_size, action_dim, bias=False)

    def forward(self, latents, condition):
        latents = self.pos_dropout(latents + self.pe[:, :latents.size(1), : ])
        latents = self.decoder(latents, memory=condition)
        return self.action_head(latents)


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1,2])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class Discriminator(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        dropout: int,
        n_layers: int,
        horizon: int,
        **kwargs,
    ) -> None:
        super().__init__()

        self.register_buffer(
            'pe',
            get_pe(hidden_size=hidden_size, max_len=2*horizon)
        )
        self.pos_dropout = nn.Dropout(p=dropout)

        self.decoder = nn.TransformerEncoder(
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=n_heads,
                dim_feedforward=4*hidden_size,
                dropout=dropout,
                activation=F.gelu,
                batch_first=True
            ),
            num_layers=n_layers
        )
        
        self.desc_head = nn.Linear(hidden_size, 1)
    
    def forward(self, latents):
        latents = self.pos_dropout(latents + self.pe[:, :latents.size(1), : ])
        latents = self.decoder(latents)

        return self.desc_head(latents)


class AutoencoderLoss(torch.nn.Module):
    def __init__(self, kl_weight=1e-6):
        super().__init__()
        self.kl_weight = kl_weight
    
    def recon_kl_loss(
        self, inputs, reconstructions, posteriors, split="train"
    ):
        rec_loss = torch.nn.functional.mse_loss(inputs, reconstructions)
        
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        loss = rec_loss + self.kl_weight * kl_loss

        log = {
            "{}/ae_total_loss".format(split): loss.clone().detach().mean(),
            "{}/kl_loss".format(split): kl_loss.detach().mean(),
            "{}/rec_loss".format(split): rec_loss.detach().mean(),
        }
        return loss, log
