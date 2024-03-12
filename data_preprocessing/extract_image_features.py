import numpy as np
import h5py
import yaml
from pathlib import Path
import tqdm
import torch
import torchvision.transforms as T
from PIL import Image 
import pickle

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from data_preprocessing.customized_r3m import R3M
import clip


DATASET_ROOT_DIR = ''
DEVICE_A = 'cuda:0'
DEVICE_B = 'cuda:1'


def get_r3m_preprocess(shorter_edge):
    return T.Compose([
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_clip_preprocess(shorter_edge):
    return T.Compose([
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])


def get_preprocess(shorter_edge):
    return T.Compose([
        T.CenterCrop(shorter_edge),
        T.Resize(224),
        T.ToTensor(),
    ])


class DatasetConverter:
    def __init__(
        self,
        dataset_name: str,
        src_h5_path: Path,
        r3m_file_path: Path,
        clip_file_path: Path,
        data_cfg: dict,
        r3m_model_id: str,
        clip_model_id: str,
    ):
        self.dataset_name = dataset_name
        self.src_h5_path = src_h5_path
        self.r3m_file_path = r3m_file_path
        self.clip_file_path = clip_file_path
        self.data_cfg = data_cfg
        self.r3m_preprocess = dict()
        self.clip_preprocess = dict()
        self.preprocess = dict()

        self.r3m = R3M(r3m_model_id).to(DEVICE_A).eval()
        self.clip, _ = clip.load(clip_model_id, device=DEVICE_B)
        self.clip = self.clip.eval()

    def run(self):
        print(f'processing {self.src_h5_path}')

        image_cfg = self.data_cfg['image']

        for view_name, shape in image_cfg.items():
            self.r3m_preprocess[view_name] = get_r3m_preprocess(min(shape[:2]))
            self.clip_preprocess[view_name] = get_clip_preprocess(min(shape[:2]))
            self.preprocess[view_name] = get_preprocess(min(shape[:2]))

        with h5py.File(str(self.src_h5_path), 'r') as src_file:
            num_episodes = int(src_file['episodes']['length'][()])
            keys = [k for k,v in image_cfg.items() if len(v) == 3 and v[-1] == 3]  # shape be like [h, w, 3]

            r3m_episodes = []
            clip_episodes = []
            for episode_index in tqdm.trange(num_episodes):
                episode = src_file['episodes'][f'episode_{episode_index}']
                r3m_episode = dict()
                clip_episode = dict()

                for key in keys:
                    src_images = episode['observation'][key][()]
                    r3m_tensors = []
                    clip_tensors = []

                    for src_image in src_images:
                        pil_img = Image.fromarray(src_image)

                        preprocessed = self.preprocess[key](pil_img)
                        r3m_image = self.r3m_preprocess[key](preprocessed)
                        r3m_tensors.append(r3m_image)
                        clip_image = self.clip_preprocess[key](preprocessed)
                        clip_tensors.append(clip_image)

                    r3m_tensors = torch.stack(r3m_tensors, dim=0).to(DEVICE_A)
                    clip_tensors = torch.stack(clip_tensors, dim=0).to(DEVICE_B)

                    r3m_features = self.r3m(r3m_tensors).cpu()
                    clip_features = self.clip.encode_image(clip_tensors).cpu()

                    r3m_episode[key] = r3m_features
                    clip_episode[key] = clip_features

                r3m_episodes.append(r3m_episode)
                clip_episodes.append(clip_episode)

            with self.r3m_file_path.open('wb') as r3m_f, self.clip_file_path.open('wb') as clip_f:
                pickle.dump(r3m_episodes, r3m_f)
                pickle.dump(clip_episodes, clip_f)


if __name__ == '__main__':
    r3m_model_id = 'resnet34'

    src_root_dir = Path(f'{DATASET_ROOT_DIR}/rt-x_h5')
    r3m_root_dir = Path(f'{DATASET_ROOT_DIR}/our_rt-x/r3m_{r3m_model_id}_image')
    r3m_root_dir.mkdir(exist_ok=True, parents=True)

    clip_model_id = 'ViT-B/32'
    clip_root_dir = Path(f'{DATASET_ROOT_DIR}/our_rt-x/clip_{clip_model_id.replace("/", "")}_image')
    clip_root_dir.mkdir(exist_ok=True, parents=True)

    with open('dataset_preprocessing/data_cfg.yaml', 'r') as f:
        data_cfgs = yaml.safe_load(f)

    for dataset_name, cfg in data_cfgs.items():
        if dataset_name[0] == '_':
            continue

        src_h5_path=src_root_dir / f'{dataset_name}.hdf5'
        r3m_file_path=r3m_root_dir / f'{dataset_name}.pkl'
        clip_file_path=clip_root_dir / f'{dataset_name}.pkl'

        if r3m_file_path.exists() and clip_file_path.exists():
            continue

        print(f'processing {dataset_name}')
        dataset_converter = DatasetConverter(
            dataset_name = dataset_name,
            src_h5_path = src_h5_path,
            r3m_file_path = r3m_file_path,
            clip_file_path = clip_file_path,
            data_cfg = cfg,
            r3m_model_id = r3m_model_id,
            clip_model_id = clip_model_id
        )
        try:
            with torch.no_grad():
                dataset_converter.run()
        except Exception as e:
            print(f'{dataset_name} error as:')
            print(e)
            print('----------------------------------------------')
        else:
            print(f'{dataset_name} done')
        torch.cuda.empty_cache()
