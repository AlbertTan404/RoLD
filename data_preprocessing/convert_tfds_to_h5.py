import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import h5py
import tqdm
import argparse
from pathlib import Path
import yaml


DATASET_ROOT_DIR = ''


def dataset_add_version(dataset_name):
    if dataset_name == 'robo_net':
        version = '1.0.0'
    elif dataset_name == 'language_table':
        version = '0.0.1'
    else:
        version = '0.1.0'
    return f'{dataset_name}/{version}'


def get_merged_dataset(src_builder):
    splits = src_builder.info.splits

    first_split = list(splits.keys())[0]
    merged_dataset = src_builder.as_dataset(split=first_split)

    for split_name in list(splits.keys())[1:]:
        dataset = src_builder.as_dataset(split=split_name)
        merged_dataset = merged_dataset.concatenate(dataset)
    
    return merged_dataset


def to_np_dtype(maybe_tf_dtype):
    dtype_map = {
        tf.bool: np.dtype('bool'),
        tf.string: str,
        tf.float16: np.float16,  
        tf.float32: np.float32,  
        tf.float64: np.float64,  
        tf.int8: np.int8,  
        tf.int16: np.int16,  
        tf.int32: np.int32,  
        tf.int64: np.int64,  
        tf.uint8: np.uint8,  
        tf.uint16: np.uint16,  
    }
    
    # keep the unfound the same
    np_dtype = dtype_map.get(maybe_tf_dtype, maybe_tf_dtype)

    return np_dtype


class DatasetConverter:
    def __init__(
        self,
        dataset_name: str,
        src_dataset_dir: Path,
        tgt_h5_path: Path):

        self.dataset_name = dataset_name

        self.src_dataset_dir = src_dataset_dir
        self.tgt_h5_path = tgt_h5_path

    def process_episode(self, h5_group: h5py.Group, episode, episode_index, shape_dtypes):
        this_episode_group = h5_group.create_group(name=f'episode_{episode_index}')

        steps = episode['steps'].as_numpy_iterator()
        steps = [step for step in steps]
        num_steps = len(steps)
        this_episode_group.create_dataset(name='length', data=num_steps)

        if 'language_instruction' in shape_dtypes.keys():
            language_instructions = [str(step['language_instruction'], encoding='utf-8') for step in steps]
        elif 'natural_language_instruction' in shape_dtypes['observation'].keys():
            language_instructions = [str(step['observation']['natural_language_instruction'], encoding='utf-8') for step in steps]
        else:
            language_instructions = ['push the T-shaped building block to the matching area']
        
        this_episode_group.create_dataset('language_instruction', data=np.array(language_instructions, dtype=h5py.string_dtype(encoding='utf-8')))

        for data_key, shape_dtype_or_dict in shape_dtypes.items():
            if 'language' in data_key:
                continue
            shape_dtype_or_dict = shape_dtypes[data_key]
            group = this_episode_group.create_group(name=data_key)

            # shape_type should be a dict, or a value
            if 'shape' in shape_dtype_or_dict.keys():
                if 'language' in data_key or shape_dtype_or_dict['dtype'] == tf.string:
                    continue
                # if shape_dtypes['action']['shape'] is not hierarchical, can directly access to step['action']
                group.create_dataset(
                    name=data_key,
                    data=self.get_episode_np_from_steps(
                        steps=steps, num_steps=num_steps, shape_dtype=shape_dtype_or_dict, data_key=data_key
                    ),
                )
            else:
                # add data hierarchicaly
                for k, shape_dtype in shape_dtype_or_dict.items():
                    if 'language' in k or shape_dtype['dtype'] == tf.string:
                        continue
                    group.create_dataset(
                        name=k,
                        data=self.get_episode_np_from_steps(
                            steps=steps, num_steps=num_steps, shape_dtype=shape_dtype, data_key=data_key, deeper_data_key=k
                        ),
                    )

    @staticmethod
    def get_episode_np_from_steps(steps, num_steps, shape_dtype, data_key, deeper_data_key=None):
        shape = shape_dtype['shape']
        dtype = shape_dtype['dtype']
        # if deeper_data_key == 'image':
        #     shape = (shape[0] // 2, shape[1] // 2, shape[2])

        episode_np_data = np.ndarray(
            shape=(num_steps,) + shape,
            dtype=to_np_dtype(dtype)
        )

        for step_index, step in enumerate(steps):
            if deeper_data_key is None:
                episode_np_data[step_index] = step[data_key]
            else:
                # if deeper_data_key == 'image':
                #     episode_np_data[step_index] =  cv2.resize(step[data_key][deeper_data_key], (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
                # else:
                episode_np_data[step_index] = step[data_key][deeper_data_key]
        
        return episode_np_data

    @staticmethod
    def dataset_info_to_dict(dataset_info: tfds.core.DatasetInfo) -> dict:
        info_dict = {
            'name': dataset_info.name,
            'version': str(dataset_info.version),
            'description': dataset_info.description,
            'homepage': dataset_info.homepage,
            'citation': dataset_info.citation,
            # 'splits': list(dataset_info.splits.keys()),
            'features': str(dataset_info.features),
        }  
        return info_dict

    def _add_dict_to_group(self, h5_group: h5py.Group, tar_dict: dict):
        for k, v in tar_dict.items():
            if isinstance(v, dict):
                g = h5_group.create_group(name=k)
                self._add_dict_to_group(g, v)
            else:
                try:
                    h5_group.create_dataset(name=k, data=v)
                except:
                    # print(f'try to convert data of {k} to string')
                    try:
                        h5_group.create_dataset(name=k, data=str(v))
                    except:
                        print(f'{k} can not be added into h5 file')

    def merge_shapes_dtypes(self, shapes, dtypes):
        assert shapes.keys() == dtypes.keys()

        res = dict()
        for k, v in shapes.items():
            if isinstance(v, dict):
                res[k] = self.merge_shapes_dtypes(shapes[k], dtypes[k])
            else:
                res[k] = {
                    'shape': shapes[k],
                    'dtype': dtypes[k]
                }
        return res

    def _add_builder_to_h5(self, h5_file: h5py.File, tfds_builder: tfds.core.dataset_builder.DatasetBuilder):
        # |__/meta_info 
        # |__/shape_info 
        # |__/episodes
        info = tfds_builder.info

        info_dict = self.dataset_info_to_dict(info)
        info_group = h5_file.create_group(name='meta_info')
        for k, v in info_dict.items():
            info_group.create_dataset(name=k, data=v)
        
        shape_type_group = h5_file.create_group(name='shape_type_info')
        shapes = info.features.shape['steps']
        dtypes = info.features.dtype['steps']
        shape_types = self.merge_shapes_dtypes(shapes, dtypes)

        self._add_dict_to_group(shape_type_group, shape_types)

        merged_dataset = get_merged_dataset(tfds_builder)
        episodes_group = h5_file.create_group('episodes')

        num_episodes = int(merged_dataset.cardinality())

        episodes_group.create_dataset(name='length', data=num_episodes)

        # single-processing
        for episode_index, episode in enumerate(tqdm.tqdm(merged_dataset, total=num_episodes)):
            self.process_episode(
                h5_group=episodes_group, episode=episode, episode_index=episode_index, shape_dtypes=shape_types 
            )

    def run(self):
        print(f'target h5 path: {self.tgt_h5_path}')

        builder = tfds.builder_from_directory(builder_dir=str(self.src_dataset_dir))
        with h5py.File(str(self.tgt_h5_path), 'w') as f:
            self._add_builder_to_h5(f, builder)

        print(f'data saved at {self.tgt_h5_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', default=0)
    args = parser.parse_args()

    dataset_index = int(args.i)

    src_root_dir = Path(f'{DATASET_ROOT_DIR}/rt-x')
    tgt_root_dir = Path(f'{DATASET_ROOT_DIR}/rt-x_h5')
    tgt_root_dir.mkdir(exist_ok=True)

    with open('./data_preprocessing/dataset_list.yaml', 'r') as f:
        dataset_list = yaml.safe_load(f)
    dataset_list = dataset_list['small']
    dataset_name = dataset_list[dataset_index]

    dataset_converter = DatasetConverter(
        dataset_name=dataset_name,
        src_dataset_dir=src_root_dir / dataset_add_version(dataset_name),
        tgt_h5_path=tgt_root_dir / f'{dataset_name}.hdf5'
    )
    dataset_converter.run()
