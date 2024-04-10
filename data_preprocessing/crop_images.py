import numpy as np
import h5py
import yaml
from pathlib import Path
import tqdm
import torch
import torchvision.transforms as T
from PIL import Image 


DATASET_ROOT_RIR = ''


def get_preprocess(shorter_edge):
    return T.Compose([
        T.CenterCrop(shorter_edge),
        T.Resize(224),
        # T.ToTensor(),
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class DatasetConverter:
    def __init__(
        self,
        dataset_name: str,
        src_h5_path: Path,
        tgt_h5_path: Path,
        data_cfg: dict
    ):
        self.dataset_name = dataset_name
        self.src_h5_path = src_h5_path
        self.tgt_h5_path = tgt_h5_path
        self.data_cfg = data_cfg
        self.preprocess = dict()

    def run(self):
        print(f'processing {self.src_h5_path}')

        image_cfg = self.data_cfg['image']

        for view_name, shape in image_cfg.items():
            self.preprocess[view_name] = get_preprocess(min(shape[:2]))

        with h5py.File(str(self.src_h5_path), 'r') as src_file, h5py.File(str(self.tgt_h5_path), 'w') as tgt_file:
            num_episodes = src_file['episodes']['length'][()]

            for episode_index in tqdm.trange(num_episodes):
                episode = src_file['episodes'][f'episode_{episode_index}']
                episode_length = int(episode['length'][()])
                tgt_episode = tgt_file.create_group(name=f'episode_{episode_index}')

                for key in image_cfg.keys():
                    src_images = episode['observation'][key][()]
                    tgt_images = np.zeros(shape=(episode_length, 224, 224, 3), dtype=np.uint8)

                    for step_idx, src_image in enumerate(src_images):
                        pil_img = Image.fromarray(src_image)
                        tgt_image = np.array(self.preprocess[key](pil_img))
                        tgt_images[step_idx] = tgt_image
                    tgt_episode.create_dataset(name=key, data=tgt_images)


if __name__ == '__main__':
    src_root_dir = Path(f'{DATASET_ROOT_DIR}/rt-x_h5')
    tgt_root_dir = Path(f'{DATASET_ROOT_DIR}/our_rt-x/cropped_images')
    tgt_root_dir.mkdir(exist_ok=True, parents=True)

    with open('RoLD/configs/data_cfg.yaml', 'r') as f:
        data_cfgs = yaml.safe_load(f)

    for dataset_name, cfg in data_cfgs.items():
        if dataset_name[0] == '_':
            continue

        src_h5_path=src_root_dir / f'{dataset_name}.hdf5'
        tgt_h5_path=tgt_root_dir / f'{dataset_name}.hdf5'

        print(f'processing {dataset_name}')
        dataset_converter = DatasetConverter(
            dataset_name=dataset_name,
            src_h5_path=src_h5_path,
            tgt_h5_path=tgt_h5_path,
            data_cfg=cfg
        )
        dataset_converter.run()
        print(f'{dataset_name} done')
