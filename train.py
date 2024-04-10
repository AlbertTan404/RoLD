import argparse
from omegaconf import OmegaConf
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.trainer import Trainer
from RoLD.utils import instantiate_from_config, get_timestamp


def get_train_val_loader(dataset, **dataloader_kwargs):
    train_ds, val_ds = dataset.split_train_val(train_ratio=0.98)
    train_loader = DataLoader(dataset=train_ds, **dataloader_kwargs, shuffle=True)
    val_loader = DataLoader(dataset=val_ds, **dataloader_kwargs, shuffle=False)
    return train_loader, val_loader


def preprocess_config(config, args):
    # set timestamp
    task = args.task
    project_name = config.model.target.split('.')[-2] + '_logs'
    config.trainer.kwargs.logger.kwargs.project = project_name
    config.trainer.kwargs.logger.kwargs.name = f'{get_timestamp()}-{task}'

    # overriding horizon
    config.horizon = args.horizon
    config.model.kwargs.model_kwargs.horizon = args.horizon
    config.dataset.kwargs.horizon = args.horizon

    # devices
    devices = args.devices
    if devices is not None:
        devices = devices.split(',')
        devices = [int(rank) for rank in devices]
        config.trainer.kwargs.devices = devices

    # avoid gpu rank overflow
    device_count = torch.cuda.device_count()
    if len(config.trainer.kwargs.devices) > device_count:
        config.trainer.kwargs.devices = list(range(device_count))
        print(f'using {device_count} devices')

    # batch size for ddp
    total_bs = config.dataloader.batch_size
    num_devices = len(config.trainer.kwargs.devices)
    bs_per_device = total_bs // num_devices
    real_bs = bs_per_device * num_devices
    if real_bs != total_bs:
        print(f'real batch size is {real_bs}')
    config.dataloader.batch_size = bs_per_device

    # dataset/tasks/mode
    data_cfg = OmegaConf.load(f'RoLD/configs/tasks/{task}_data_cfg.yaml')
    if task == 'rt-x':
        config.model.kwargs.mode = 'pretraining'
        config.dataset.kwargs.get_image_dict = False
        config.dataset.kwargs.get_canonical_image = True
        config.trainer.kwargs.max_epochs = config.trainer.kwargs.pretrain_max_epochs
        config.trainer.kwargs.pop('pretrain_max_epochs')
        config.dataset.kwargs.get_low_dim = False
    else:
        config.model.kwargs.mode = 'finetuning'
        config.trainer.kwargs.pop('pretrain_max_epochs')
        config.dataset.kwargs.get_image_dict = True
        config.dataset.kwargs.get_canonical_image = False
        assert 'low_dim' in data_cfg.keys()
        low_dim_feature_dim = sum([dim for dim in data_cfg.low_dim.values()])
        config.model.kwargs.model_kwargs.low_dim_feature_dim = low_dim_feature_dim
        config.dataset.kwargs.get_low_dim = True

    datasets_cfg = data_cfg.datasets
    config.dataset.kwargs.root_dir = Path(f'~/data/data/RoLD/our_{task}').expanduser()
    config.dataset.kwargs.data_cfg = datasets_cfg
    config.dataset.kwargs.dataset_names = [key for key in datasets_cfg.keys() if key[0] != '_' and '_mh' not in key]
    config.dataset.kwargs.average_step_per_episode = data_cfg.average_step_per_episode

    # feature dimension:
    if config.dataset.kwargs.feature_type[:3] == 'r3m':
        config.model.kwargs.model_kwargs.language_feature_dim = 768
    else:  # clip
        config.model.kwargs.model_kwargs.language_feature_dim = 512

    return config


def get_parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config_name',
        default='downsample_cvae'
    )
    parser.add_argument(
        '--task',
        choices=['robomimic_ph', 'robomimic_mh', 'rt-x', 'meta-world'],
        default='robomimic_ph' # rt-x or robomimic or meta_world
    )
    parser.add_argument(
        '--devices',
        type=str,
        default='0',
    )
    parser.add_argument(
        '--horizon',
        type=int,
        default=16
    )

    return parser.parse_args()


def main():
    args = get_parser_args()

    raw_config = OmegaConf.load(f'RoLD/configs/{args.config_name}.yaml')
    OmegaConf.resolve(raw_config)
    config = preprocess_config(raw_config, args)

    pl.seed_everything(config.seed)

    model: pl.LightningModule = instantiate_from_config(config.model, extra_kwargs={"all_config": config})

    dataset = instantiate_from_config(config.dataset)
    train_loader, val_loader = get_train_val_loader(dataset=dataset, **config.dataloader)

    epoch_length = len(train_loader) // len(config.trainer.kwargs.devices)
    config.model.kwargs.training_kwargs['num_training_steps'] = epoch_length * config.trainer.kwargs.max_epochs

    trainer: Trainer = instantiate_from_config(config.trainer)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()
