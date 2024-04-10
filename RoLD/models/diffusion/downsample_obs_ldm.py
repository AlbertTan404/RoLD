import copy
from typing import Any
from collections import OrderedDict
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.schedulers import DDPMScheduler, DDIMScheduler

import lightning.pytorch as pl

from RoLD.models.common import SinusoidalPosEmb, get_pe, WrappedTransformerEncoder, WrappedTransformerDecoder, ResBottleneck, ImageAdapter
from RoLD.models.autoencoder.downsample_cvae import DownsampleCVAE


class DownsampleObsLDM(pl.LightningModule):
    def __init__(
        self,
        ae_ckpt_path,
        model_kwargs,
        training_kwargs,
        noise_scheduler_kwargs,
        mode,
        all_config=None
    ) -> None:
        super().__init__()
        # three modes: pretraining, finetuning, inference

        ckpt_path = model_kwargs.ckpt_path
        if ckpt_path is not None:
            assert mode == 'finetuning' or mode == 'inference'
            ckpt = torch.load(ckpt_path)
            hyper_params = copy.deepcopy(ckpt['hyper_parameters'])
            low_dim_feature_dim = model_kwargs.low_dim_feature_dim
            model_kwargs = hyper_params['model_kwargs']
            model_kwargs.low_dim_feature_dim = low_dim_feature_dim

        # initialze model
        self.training_kwargs = training_kwargs
        self.model_kwargs = model_kwargs
        self.noise_scheduler_kwargs= noise_scheduler_kwargs
        self.save_hyperparameters()
 
        # initialze autoencoder
        ae_ckpt = torch.load(ae_ckpt_path)
        ae_config = ae_ckpt['hyper_parameters']
        if mode == 'finetuning' and ae_config['mode'] == 'pretraining':
            raise ValueError('you should load a finetuned AE during finetuning ldm')
        ae_config['model_kwargs']['ckpt_path'] = ae_ckpt_path
        autoencoder = DownsampleCVAE(**ae_config)  # init includes loading ckpt
        # freeze autoencoder
        for p in autoencoder.parameters():
            p.requires_grad = False
        del ae_ckpt

        self.hidden_size = hidden_size = model_kwargs['hidden_size'] = autoencoder.hidden_size
        self.latent_size = latent_size = model_kwargs['latent_size'] = autoencoder.latent_size
        self.horizon = horizon = model_kwargs['horizon']

        self.time_emb = SinusoidalPosEmb(dim=hidden_size)
        self.register_buffer(
            'pe', get_pe(hidden_size=hidden_size, max_len=horizon*2)
        )

        self.z_up = nn.Linear(latent_size, hidden_size)
        self.denoiser = WrappedTransformerEncoder(**model_kwargs)
        self.z_down = nn.Linear(hidden_size, latent_size)

        if self.noise_scheduler_kwargs.get('num_inference_timesteps') is not None:
            self.num_inference_timesteps = self.noise_scheduler_kwargs.pop('num_inference_timesteps')
        self.noise_scheduler = DDPMScheduler(**self.noise_scheduler_kwargs)

        self.language_emb = nn.Linear(in_features=model_kwargs['language_feature_dim'], out_features=hidden_size)

        if ckpt_path is not None:
            # attach obs, then load real params
            # copy pretrained img_emb
            if hasattr(autoencoder, 'img_emb'):
                self.img_emb = copy.deepcopy(autoencoder.img_emb)
            else:
                self.img_emb = ResBottleneck(hidden_size=hidden_size)
            for p in self.img_emb.parameters():
                p.requires_grad = True
            if model_kwargs.get('low_dim_feature_dim') is not None:
                assert mode == 'finetuning' or mode == 'inference'
                # copy pretrained low_dim_emb
                if hasattr(autoencoder, 'low_dim_emb'):
                    self.low_dim_emb = copy.deepcopy(autoencoder.low_dim_emb)
                else:
                    self.low_dim_emb = nn.Linear(model_kwargs['low_dim_feature_dim'], hidden_size)
                for p in self.low_dim_emb.parameters():
                    p.requires_grad = True
            else:
                assert mode == 'pretraining'
                self.low_dim_emb = None

            self.load_state_dict(state_dict=ckpt['state_dict'], strict=False)  # only load the ldm part
            del ckpt
            print(f'WARNING: ignoring LDM config, LDM loaded from {ckpt_path}')
        else:
            # apply init on other params
            self.apply(self._init_weights)
            # then attach obs 
            # copy pretrained img_emb
            if hasattr(autoencoder, 'img_emb'):
                self.img_emb = copy.deepcopy(autoencoder.img_emb)
            else:
                self.img_emb = ResBottleneck(hidden_size=hidden_size)
            for p in self.img_emb.parameters():
                p.requires_grad = True
            if model_kwargs.get('low_dim_feature_dim') is not None:
                assert mode == 'finetuning' or mode == 'inference'
                # copy pretrained low_dim_emb
                if hasattr(autoencoder, 'low_dim_emb'):
                    self.low_dim_emb = copy.deepcopy(autoencoder.low_dim_emb)
                else:
                    self.low_dim_emb = nn.Linear(model_kwargs['low_dim_feature_dim'], hidden_size)
                for p in self.low_dim_emb.parameters():
                    p.requires_grad = True
            else:
                assert mode == 'pretraining'
                self.low_dim_emb = None

        # must attach autoencoder at last to avoid loading params in ldm ckpt or init weights
        self.autoencoder = autoencoder
        self.last_training_batch = None
        
    def configure_optimizers(self):
        kwargs = self.training_kwargs

        tuned_parameters = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.Adam(
            tuned_parameters,
            lr=kwargs.lr,
        )
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=kwargs.warmup_steps, num_training_steps=kwargs.num_training_steps)

        self.lr_scheduler = scheduler
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }

    def _init_weights(self, module):
        ignore_types = (nn.Dropout, 
            SinusoidalPosEmb, 
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
            WrappedTransformerDecoder,
            WrappedTransformerEncoder,
            nn.LeakyReLU,
            ResBottleneck,
            DownsampleObsLDM,
            DownsampleCVAE  # double check
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def pred_epsilon(self, noise, timestep, language_emb, ldm_obs_emb):
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None]
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(noise.shape[0]).to(device=noise.device)
        time_emb = self.time_emb(timesteps).unsqueeze(1)

        embeddings = torch.cat([time_emb, language_emb, ldm_obs_emb, self.z_up(noise)], dim=1)
        pred_noise = self.denoiser(embeddings)
        return self.z_down(pred_noise[:, -1:, :])

    def get_language_emb(self, raw_language_features):
        return self.language_emb(raw_language_features)

    def get_obs_emb(self, raw_image_features, raw_low_dim_data):
        image_emb = self.img_emb(raw_image_features)
        if raw_low_dim_data is not None and self.low_dim_emb is not None:
            low_dim_emb = self.low_dim_emb(raw_low_dim_data)
            return torch.cat([image_emb, low_dim_emb], dim=1)
        else:
            return image_emb

    def predict_action(self, raw_language_features, raw_image_features, raw_low_dim_data=None):
        # scheduler = DDIMScheduler.from_config(self.noise_scheduler.config)
        # scheduler.set_timesteps(self.num_inference_timesteps)
        scheduler = self.noise_scheduler

        language_emb = self.get_language_emb(raw_language_features=raw_language_features)
        ldm_obs_emb = self.get_obs_emb(raw_image_features=raw_image_features, raw_low_dim_data=raw_low_dim_data)
        ae_obs_emb = self.autoencoder.get_obs_emb(raw_image_features=raw_image_features, raw_low_dim_data=raw_low_dim_data)
        batch_size = language_emb.shape[0]

        # recover z
        z = torch.randn(size=(batch_size, 1, self.latent_size), dtype=language_emb.dtype, device=language_emb.device)
        for t in scheduler.timesteps:
            model_output = self.pred_epsilon(
                noise=z, timestep=t, language_emb=language_emb, ldm_obs_emb=ldm_obs_emb)
            z = scheduler.step(model_output, t, z).prev_sample
        
        pred_action = self.autoencoder.decode(obs_emb=ae_obs_emb, z=z, raw_language_features=raw_language_features)
        return pred_action

    def forward(self, batch, batch_idx, split='train'):
        # autoencoder
        posterior, _ = self.autoencoder.encode(batch)
        ldm_obs_emb = self.get_obs_emb(raw_image_features=batch['image'], raw_low_dim_data=batch.get('low_dim'))
        z = posterior.sample()
        language_emb = self.get_language_emb(batch['language'])

        # diffusion
        noise = torch.randn(z.shape, device=z.device)
        timesteps = torch.randint(
            low=0, high=self.noise_scheduler.config.num_train_timesteps, size=(z.shape[0],), device=z.device
        ).long()
        noisy_latent = self.noise_scheduler.add_noise(z, noise, timesteps)
 
        pred = self.pred_epsilon(
            noise = noisy_latent,
            timestep = timesteps,
            language_emb = language_emb,
            ldm_obs_emb = ldm_obs_emb
        )
        denoise_loss = F.mse_loss(noise, pred)

        return {f'{split}/denoise_loss': denoise_loss}

    def training_step(self, batch, batch_idx):
        self.last_training_batch = batch
        forward_results = self.forward(batch=batch, batch_idx=batch_idx, split='train')
        self.log_dict(forward_results, sync_dist=True)
        return forward_results['train/denoise_loss']

    # def on_train_epoch_end(self) -> None:
    #     with torch.no_grad():
    #         batch = self.last_training_batch
    #         raw_action = batch['action']
    #         pred_action = self.predict_action(
    #             raw_language_features=batch['language'],
    #             raw_image_features=batch['image'],
    #             raw_low_dim_data=batch.get('low_dim')
    #         )
    #         model_mse_error = F.mse_loss(raw_action, pred_action)
    #         self.log('train/model_mse_error', model_mse_error, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        forward_results = self.forward(batch=batch, batch_idx=batch_idx, split='val')
        self.log_dict(forward_results, sync_dist=True)

        if batch_idx == 0:
            raw_action = batch['action']
            pred_action = self.predict_action(
                raw_language_features=batch['language'],
                raw_image_features = batch['image'],
                raw_low_dim_data=batch.get('low_dim')
            )
            model_mse_error = F.mse_loss(raw_action, pred_action)
            self.log('val/model_mse_error', model_mse_error, sync_dist=True)
        return forward_results['val/denoise_loss']
