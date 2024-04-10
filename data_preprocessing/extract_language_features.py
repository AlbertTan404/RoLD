import sys
import os
import numpy as np
import h5py
import yaml
from pathlib import Path
import tqdm
import pickle
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import clip
from data_preprocessing.customized_r3m import r3m


DATASET_ROOT_DIR = ''
DEVICE_A = 'cuda:0'
DEVICE_B = 'cuda:1'


def are_elements_same_along_first_dimension(data):  
    all_same = np.all(np.equal(data[:-1], data[1:]), axis=0)  
    return np.all(all_same)


class DatasetConverter:
    def __init__(
        self,
        dataset_name: str,
        src_h5_path: Path,
        r3m_file_path: Path,
        clip_file_path: Path,
        data_cfg: dict,
        clip_model_id: str,
    ):
        self.dataset_name = dataset_name
        self.src_h5_path = src_h5_path
        self.r3m_file_path = r3m_file_path
        self.clip_file_path = clip_file_path
        self.data_cfg = data_cfg

        self.r3m_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.r3m_model = AutoModel.from_pretrained("distilbert-base-uncased").to(DEVICE_A)

        self.clip, _ = clip.load(clip_model_id, device=DEVICE_B)
        self.clip = self.clip.eval()

    def extract_r3m_features(self, langs):
        with torch.no_grad():
            encoded_input = self.r3m_tokenizer(langs, return_tensors='pt', padding=True)
            input_ids = encoded_input['input_ids'].to(DEVICE_A)
            attention_mask = encoded_input['attention_mask'].to(DEVICE_A)
            lang_embedding = self.r3m_model(input_ids, attention_mask=attention_mask).last_hidden_state
            lang_embedding = lang_embedding.mean(1)
        return lang_embedding.cpu()
    
    def extract_clip_features(self, langs):
        with torch.no_grad():
            text = clip.tokenize(langs).to(DEVICE_B)
            text_features = self.clip.encode_text(text)
        return text_features.cpu()
    
    def all_the_same(self):
        outer_key = self.data_cfg['language']['outer_key']
        inner_key = self.data_cfg['language']['inner_key']

        with h5py.File(str(self.src_h5_path), 'r') as src_file:
            num_episodes = src_file['episodes']['length'][()]

            for episode_index in tqdm.trange(num_episodes):
                episode = src_file['episodes'][f'episode_{episode_index}']
                if inner_key is None:
                    episode = episode[outer_key][::5]  # skip some frames to reduce computation
                else:
                    episode = episode[outer_key][inner_key][::10]
                sameness = are_elements_same_along_first_dimension(episode)
                if not sameness:
                    print(f'sentences in episode_{episode_index} are not all the same')
                    return False
        return True
    
    def run(self):
        if self.all_the_same():
            self.run_all_same()
    
    def run_step_by_step(self):
        print(f'processing {self.src_h5_path}')

        outer_key = self.data_cfg['language']['outer_key']
        inner_key = self.data_cfg['language']['inner_key']

        with h5py.File(str(self.src_h5_path), 'r') as src_file:
            num_episodes = src_file['episodes']['length'][()]
            episodes = []

            for episode_index in tqdm.trange(num_episodes):
                episode = src_file['episodes'][f'episode_{episode_index}']
                if inner_key is None:
                    episode = episode[outer_key][()]
                else:
                    episode = episode[outer_key][inner_key][()]
                episodes.append([s.decode() for s in episode])

        r3m_data = []
        clip_data = []    
        for langs in episodes:
            r3m_features = self.extract_r3m_features(langs)
            clip_fetures = self.extract_clip_features(langs)
            r3m_data.append(r3m_features)
            clip_data.append(clip_fetures)

        with self.r3m_file_path.open('wb') as f:
            pickle.dump(r3m_data, f)
        with self.clip_file_path.open('wb') as f:
            pickle.dump(clip_data, f)

    def run_all_same(self):
        print(f'processing {self.src_h5_path}')

        outer_key = self.data_cfg['language']['outer_key']
        inner_key = self.data_cfg['language']['inner_key']

        with h5py.File(str(self.src_h5_path), 'r') as src_file:
            num_episodes = src_file['episodes']['length'][()]
            sentences = []

            for episode_index in tqdm.trange(num_episodes):
                episode = src_file['episodes'][f'episode_{episode_index}']
                if inner_key is None:
                    episode = episode[outer_key][0]
                else:
                    episode = episode[outer_key][inner_key][0]
                sentences.append(episode.decode())

        begin_idx = 0
        batch_size = 2048
        r3m_data = []
        clip_data = []
        while True:
            end_idx = min(begin_idx + batch_size, num_episodes)

            r3m_features = self.extract_r3m_features(sentences[begin_idx: end_idx])
            clip_fetures = self.extract_clip_features(sentences[begin_idx: end_idx])
            r3m_data.append(r3m_features)
            clip_data.append(clip_fetures)

            begin_idx = end_idx
            if begin_idx >= num_episodes:
                break
            print(f'{end_idx}', end=' ')
        
        r3m_data = torch.cat(r3m_data, dim=0)
        clip_data = torch.cat(clip_data, dim=0)

        with self.r3m_file_path.open('wb') as f:
            pickle.dump(r3m_data, f)
        with self.clip_file_path.open('wb') as f:
            pickle.dump(clip_data, f)


if __name__ == '__main__':
    src_root_dir = Path(f'{DATASET_ROOT_DIR}/rt-x_h5')

    clip_model_id = 'ViT-B/32'

    r3m_root_dir = Path(f'{DATASET_ROOT_DIR}/our_rt-x/distilbert_language')
    clip_root_dir = Path(f'{DATASET_ROOT_DIR}/our_rt-x/clip_{clip_model_id.replace("/", "")}_language')

    r3m_root_dir.mkdir(exist_ok=True, parents=True)
    clip_root_dir.mkdir(exist_ok=True, parents=True)

    with open('RoLD/configs/data_cfg.yaml', 'r') as f:
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
            dataset_name=dataset_name,
            src_h5_path=src_h5_path,
            r3m_file_path=r3m_file_path,
            clip_file_path=clip_file_path,
            data_cfg=cfg,
            clip_model_id=clip_model_id
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
