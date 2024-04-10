import os
import numpy as np
import h5py
import yaml
from pathlib import Path
from PIL import Image 
import time
from concurrent.futures import ProcessPoolExecutor


DATASET_ROOT_DIR = ''


def process_episode(dataset_dir: Path, h5_path, episode_path, episode_index, obs_keys):
    with h5py.File(h5_path, 'r') as f:
        episodes = f[episode_path]
        episode_dir = dataset_dir / f'episode_{episode_index}'
        episode_dir.mkdir(exist_ok=True)
        obs_data = episodes[f'episode_{episode_index}']['observation']
        num_steps = episodes[f'episode_{episode_index}']['length'][()]

        for obs_key in obs_keys:
            obs_dir = episode_dir / obs_key
            obs_dir.mkdir(exist_ok=True)
            images = obs_data[obs_key][()]
            for step_index in range(num_steps):
                np_image = images[step_index]
                if 'depth' in obs_key:
                    np_image = (np_image - np_image.min()) / (np_image.max() - np_image.min()) * 65535
                    np_image = np.squeeze(np_image.astype(np.uint16))
                image = Image.fromarray(np_image)
                image.save(str(obs_dir / f'{step_index}.png'))


class DatasetConverter:
    def __init__(
        self,
        dataset_name: str,
        src_h5_path: Path,
        temp_h5_path: Path,
        tgt_png_dir: Path):

        self.dataset_name = dataset_name

        self.src_h5_path = src_h5_path
        self.temp_h5_path = temp_h5_path
        self.tgt_png_dir = tgt_png_dir
        self.tgt_png_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        print(f'processing {self.src_h5_path}')
        with h5py.File(str(self.src_h5_path), 'r') as f:
            obs_info = f['shape_type_info']['observation']

            obs_keys = []
            for key in obs_info.keys():
                key_shape = obs_info[key].get('shape')
                if key_shape is not None and key_shape[()].shape[0] == 3 and 'flow' not in key:
                    obs_keys.append(key)

            num_episodes = f['episodes']['length'][()]
        episode_indices = list(range(num_episodes))
        print(f'num of episodes: {num_episodes}')
        begin_time = time.perf_counter()

        with ProcessPoolExecutor() as ppe:
            list(ppe.map(process_episode, [self.tgt_png_dir]*num_episodes, [self.src_h5_path]*num_episodes, ['episodes']*num_episodes, episode_indices, [obs_keys] * num_episodes))

        print(f'data saved at {self.tgt_png_dir}, took {time.perf_counter() - begin_time} seconds')
    
    def copy_group(self, src_group, dst_group):  
        for key in src_group:  
            src_obj = src_group[key]  
            if isinstance(src_obj, h5py.Dataset):  
                if key not in self.obs_keys:
                    src_group.copy(key, dst_group, key)  
            elif isinstance(src_obj, h5py.Group):  
                dst_sub_group = dst_group.create_group(key)  
                self.copy_group(src_obj, dst_sub_group)
    
    def reclaiming(self):
        with h5py.File(str(self.src_h5_path), 'r') as f:
            obs_info = f['shape_type_info']['observation']

            obs_keys = []
            for key in obs_info.keys():
                key_shape = obs_info[key].get('shape')
                if key_shape is not None and key_shape[()].shape[0] == 3 and 'flow' not in key:
                    obs_keys.append(key)
        self.obs_keys = obs_keys

        with h5py.File(str(self.src_h5_path), "r") as src_file, h5py.File(str(self.temp_h5_path), "w") as dst_file:  
            self.copy_group(src_file, dst_file)


if __name__ == '__main__':
    src_root_dir = Path(f'{DATASET_ROOT_DIR}/rt-x_h5')
    temp_root_dir = Path(f'{DATASET_ROOT_DIR}/_rt-x_h5')
    temp_root_dir.mkdir(exist_ok=True)
    tgt_root_dir = Path(f'{DATASET_ROOT_DIR}/rt-x_png')
    tgt_root_dir.mkdir(exist_ok=True)

    with open('./data_preprocessing/dataset_list.yaml', 'r') as f:
        dataset_list = yaml.safe_load(f)
    dataset_list = dataset_list['large'] + dataset_list['small']

    for dataset_name in dataset_list:
        src_h5_path=src_root_dir / f'{dataset_name}.hdf5'
        temp_h5_path=temp_root_dir / f'{dataset_name}.hdf5'
        tgt_png_dir=tgt_root_dir / dataset_name

        if temp_h5_path.exists():
            continue
        dataset_converter = DatasetConverter(
            dataset_name=dataset_name,
            src_h5_path=src_h5_path,
            temp_h5_path=temp_h5_path,
            tgt_png_dir=tgt_png_dir
        )
        dataset_converter.run()
        dataset_converter.reclaiming()
        # os.remove(str(dataset_converter.src_h5_path))
        print(f'{dataset_name} done')
