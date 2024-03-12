import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import numpy as np
import h5py
import yaml
from pathlib import Path
import tqdm
import pickle
import torch
from concurrent.futures import ProcessPoolExecutor as PPE
from data_preprocessing.utils import uniform_normalization, scale_only_normalization


DATASET_ROOT_DIR = ''


class DatasetConverter:
    def __init__(
        self,
        dataset_name: str,
        src_h5_path: Path,
        normalized_pkl_path: Path,
        raw_pkl_path: Path,
        data_cfg: dict
    ):
        self.dataset_name = dataset_name
        self.src_h5_path = src_h5_path
        self.normalized_pkl_path = normalized_pkl_path
        self.raw_pkl_path = raw_pkl_path
        self.data_cfg = data_cfg

    def run(self):
        print(f'processing {self.src_h5_path}')

        with h5py.File(str(self.src_h5_path), 'r') as src_file:
            num_episodes = src_file['episodes']['length'][()]
            raw_episodes = []
            normalized_episodes = []

            action_cfg = self.data_cfg['action']
            action_outer_key = action_cfg['outer_key']
            action_inner_keys = action_cfg['inner_keys']
            index_mapping = action_cfg['index_mapping']
            mins = np.array(action_cfg['min'], dtype=np.float32)
            maxs = np.array(action_cfg['max'], dtype=np.float32)

            for episode_index in tqdm.trange(num_episodes):
                episode = src_file['episodes'][f'episode_{episode_index}']

                length = int(episode['length'][()])
                concated_data = \
                    [np.zeros(shape=(length, 1), dtype=np.float32)] +\
                    [episode[action_outer_key][a_inner_key][()] for a_inner_key in action_inner_keys]
                for data_idx, d in enumerate(concated_data):
                    if len(d.shape) == 1:
                        concated_data[data_idx] = np.expand_dims(d, axis=1)
                    elif d.dtype == bool:
                        concated_data[data_idx] = np.array(d, dtype=np.float32).reshape(-1, 1)
                concated_data = np.concatenate(concated_data, axis=-1)

                arranged_action = np.zeros(shape=(length, 7), dtype=np.float32)
                for tgt_idx, src_idx in enumerate(index_mapping):
                    arranged_action[:, tgt_idx] = concated_data[:, src_idx]

                raw_episodes.append(torch.from_numpy(arranged_action).to(dtype=torch.float32))

                normalized_action = scale_only_normalization(arranged_action, min_values=mins, max_values=maxs)
                if not action_cfg['gripper_close_is_positive']:
                    normalized_action[:, -1] = -normalized_action[:, -1]
                normalized_episodes.append(torch.from_numpy(normalized_action))

        with self.raw_pkl_path.open('wb') as f:
            pickle.dump(raw_episodes, f)
        with self.normalized_pkl_path.open('wb') as f:
            pickle.dump(normalized_episodes, f)


def process_dataset(dataset_name, src_h5_path, normalized_pkl_path, raw_pkl_path, data_cfg):
    dataset_converter = DatasetConverter(
        dataset_name=dataset_name,
        src_h5_path=src_h5_path,
        normalized_pkl_path=normalized_pkl_path,
        raw_pkl_path=raw_pkl_path,
        data_cfg=data_cfg
    )
    try:
        dataset_converter.run()
    except Exception as e:
        print(f'{dataset_name} error as:')
        print(e)
        print('----------------------------------------------')
    else:
        print(f'{dataset_name} done')


if __name__ == '__main__':
    src_root_dir = Path(f'{DATASET_ROOT_DIR}/rt-x_h5')
    raw_root_dir = Path(f'{DATASET_ROOT_DIR}/rt-x_pt/raw_actions')
    normalized_root_dir = Path(f'{DATASET_ROOT_DIR}/rt-x_pt/normalized_actions')
    raw_root_dir.mkdir(exist_ok=True, parents=True)
    normalized_root_dir.mkdir(exist_ok=True, parents=True)

    with open('./data_preprocessing/rt-x_data_cfg.yaml', 'r') as f:
        data_cfgs = yaml.safe_load(f)['datasets']

    dataset_name_list = [n for n in data_cfgs.keys() if n[0] != '_']
    dataset_cfg_list = [data_cfgs[n] for n in dataset_name_list]
    src_h5_path_list = [src_root_dir / f'{n}.hdf5' for n in dataset_name_list]
    normalized_pkl_path_list = [normalized_root_dir / f'{n}.pkl' for n in dataset_name_list]
    raw_pkl_path_list = [raw_root_dir / f'{n}.pkl' for n in dataset_name_list]

    with PPE() as ppe:
        list(ppe.map(process_dataset, dataset_name_list, src_h5_path_list, normalized_pkl_path_list, raw_pkl_path_list, dataset_cfg_list))
