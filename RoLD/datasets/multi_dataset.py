import bisect
from itertools import accumulate
import os
import copy
import random
from typing import List
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset


class SingleDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        action_path: str,
        language_path: str,
        image_path: str,
        low_dim_path: str,
        data_cfg: dict,
        load_language: bool,
        load_image: bool,
        load_low_dim: bool
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.data_cfg = data_cfg

        self.action_path = action_path
        self.language_path = language_path
        self.image_path = image_path
        self.low_dim_path = low_dim_path

        self.image_cfg = data_cfg['image']
        try:
            self.image_keys = list(data_cfg['image'].keys())
        except:
            pass
        self.canonical_image_key = data_cfg['canonical_view']

        self.image_preprocess = None
        self.action_data = None
        self.image_data = None
        self.language_data = None
        self.load_image = load_image
        self.load_language = load_language
        self.load_low_dim = load_low_dim

    def split_train_val(self, train_ratio=0.98):
        val_ds = copy.deepcopy(self)

        with open(self.action_path, 'rb') as f:
            action_data = pickle.load(f)
        n_episodes = len(action_data)
        if n_episodes > 10:
            num_train_episodes = int(train_ratio * len(action_data)) - 1 # at least one val episode
        else:
            num_train_episodes = len(action_data) - 1 # few shot, ignoring val
        self.action_data = action_data[:num_train_episodes]
        val_ds.action_data = action_data[num_train_episodes:]

        if self.load_language:
            with open(self.language_path, 'rb') as f:
                language_data = pickle.load(f)
            if len(language_data) == 1:
                self.language_data = language_data
                val_ds.language_data = language_data
            else:
                self.language_data = language_data[:num_train_episodes]
                val_ds.language_data = language_data[num_train_episodes:]

        if self.load_image:
            with open(self.image_path, 'rb') as f:
                image_data = pickle.load(f)
            self.image_data = image_data[:num_train_episodes]
            val_ds.image_data = image_data[num_train_episodes:]
        
        if self.load_low_dim:
            with open(self.low_dim_path, 'rb') as f:
                low_dim_data = pickle.load(f)
            self.low_dim_data = low_dim_data[:num_train_episodes]
            val_ds.low_dim_data = low_dim_data[num_train_episodes:]

        return self, val_ds

    def __len__(self):
        # return num of episodes
        return len(self.action_data)

    def get_data(
        self, horizon, episode_index, get_language, get_canonical_image, get_image_dict, get_low_dim,
        recursize_depth=0, 
    ):
        if recursize_depth > 100:
            print(f'No enough episode longer than {horizon} steps in {self.dataset_name}')
            return None

        num_steps = len(self.action_data[episode_index])
        if num_steps < horizon:  # get random one
            episode_index = random.randint(0, len(self) - 1)
            return self.get_data(
                horizon=horizon, episode_index=episode_index, get_language=get_language, get_canonical_image=get_canonical_image, get_image_dict=get_image_dict, get_low_dim=get_low_dim,
                recursize_depth = recursize_depth + 1, 
            )
        
        # actions
        begin_index = random.randint(0, num_steps - horizon)
        action_seq = self.action_data[episode_index][begin_index: begin_index + horizon]

        data = {'action': action_seq}

        if get_canonical_image:
            key = random.choice(self.image_keys)
            image_feature = self.image_data[episode_index][key][begin_index].unsqueeze(0)
            data.update({'image': image_feature.to(dtype=torch.float32)})

        elif get_image_dict:
            # check all the data have same num of views in the same dataset
            multiview_image_tensors = []
            for key in sorted(self.image_cfg):  # must be sorted
                image_feature = self.image_data[episode_index][key][begin_index]
                multiview_image_tensors.append(image_feature)
            multiview_image_tensors = torch.stack(multiview_image_tensors, dim=0).to(dtype=torch.float32)
                
            data.update({'image': multiview_image_tensors})

        if get_language:
            if len(self.language_data) == 1:
                # downstream finetuning dataset
                language_feature = self.language_data[0]
            else:
                language_feature = self.language_data[episode_index]
            data.update({'language': language_feature.unsqueeze(0).to(dtype=torch.float32)})
        
        if get_low_dim:
            data.update({'low_dim': self.low_dim_data[episode_index][begin_index].unsqueeze(0)})
        
        return data


class MultiDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        dataset_names: List[str],
        horizon: int,
        data_cfg: dict,
        get_language: bool,
        get_canonical_image: bool,
        get_image_dict: bool,
        get_low_dim: bool,
        feature_type: str,
        average_step_per_episode: int
    ):
        super().__init__()
        self.horizon = horizon

        self.get_language = get_language
        self.get_canonical_image = get_canonical_image
        self.get_image_dict = get_image_dict
        self.get_low_dim = get_low_dim
        self.average_step_per_episode = average_step_per_episode 

        self.dataset_names = dataset_names
        self.datasets = [
            SingleDataset(
                dataset_name = dataset_name,
                action_path = os.path.join(root_dir, 'normalized_actions', f'{dataset_name}.pkl'),
                low_dim_path = os.path.join(root_dir, f'low_dim', f'{dataset_name}.pkl'),
                language_path = os.path.join(root_dir, f'{feature_type}_language', f'{dataset_name}.pkl'),
                image_path = os.path.join(root_dir, f'{feature_type}_image', f'{dataset_name}.pkl'),
                data_cfg = data_cfg[dataset_name],
                load_language = get_language,
                load_image = get_canonical_image or get_image_dict,
                load_low_dim = get_low_dim
            ) for dataset_name in dataset_names
        ]

        self.dynamic_variables_loaded = False
        self.dataset_lengthes = None
        self.accumulated_lengthes = None
        self.image_preprocess = None

    def register_image_preprocess_hook(self, func):
        self.image_preprocess = func

    def split_train_val(self, train_ratio):
        train_datasets = []
        val_datasets = []

        for dataset in self.datasets:
            train_ds, val_ds = dataset.split_train_val(train_ratio)
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)

        val_dataset = copy.deepcopy(self)

        val_dataset.datasets = val_datasets
        self.datasets = train_datasets

        self.load_dynamic_variables()
        val_dataset.load_dynamic_variables()
        print(f'{len(train_datasets)} datasets loaded')
        return self, val_dataset

    def load_dynamic_variables(self):
        # called after splitting
        self.dataset_lengthes = [len(dataset) for dataset in self.datasets]
        self.accumulated_lengthes = list(accumulate(self.dataset_lengthes))
        self.dynamic_variables_loaded = True

    def __len__(self):
        if not self.dynamic_variables_loaded:
            raise ValueError('please call load_dynamic_variables() before training')
        # len(dataset) returns the num of episodes, scaling by an avarage number of steps per episode
        return self.accumulated_lengthes[-1] * self.average_step_per_episode
    
    def get_dataset_and_episode_index(self, index):
        if not self.dynamic_variables_loaded:
            raise ValueError('please call load_dynamic_variables() before training')

        index = index % self.accumulated_lengthes[-1]

        dataset_idx = bisect.bisect_right(self.accumulated_lengthes, index)

        if dataset_idx == 0:
            data_index = index
        else:
            data_index = index - self.accumulated_lengthes[dataset_idx - 1]

        return dataset_idx, data_index

    def __getitem__(self, index):
        dataset_index, data_index = self.get_dataset_and_episode_index(index)
        return self.datasets[dataset_index].get_data(
            horizon = self.horizon,
            episode_index = data_index,
            get_canonical_image = self.get_canonical_image,
            get_image_dict = self.get_image_dict,
            get_language = self.get_language,
            get_low_dim = self.get_low_dim
        )
