# Multi-task Manipulation Policy Modeling with Visuomotor Latent Diffusion
#### Wenhui Tan, Bei Liu, Junbo Zhang, Ruihua Song, Jianlong Fu

---

## Data Preprocessing:

### Dataset access:
Refer to the [rt-x official repo](https://github.com/google-deepmind/open_x_embodiment#dataset-access)

### Usages:
- Convert tfds to h5 file with *convert_tfds_to_h5.py*  # converting large datasets takes MASSIVE disk space. (up to 8 TB for kuka)

- Visualize the datasets with the processed h5 file with *check_data.ipynb*.

- Extract raw images with *move_h5_image_to_png.py*.

- Extract image and language features for most efficient policy model training with *extract_language_features.py* and *extract_image_features.py*. (we use R3M and CLIP, and it's easy for you to customize it)

- Normalize actions according to the statistics for unified training with *normalize_actoins.py* and *rt-x_data_cfg.yaml*. 

### Features:
- For extracting features, we use multi-processing among datasets for better efficiency.
- For converting tfds to hdf5 files, we also support parallel processing by setting the --index option.  # default turned off


### Environment
In your python environment:

- install basic libraries
```
pip install tensorflow tensorflow-datasets
```

```
conda install h5py yaml jupyter tqdm omegaconf gdown matplotlib
```

- install pytorch (version not strictly restricted)
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

```
pip install lightning transformers diffusers
```

- optional: install clip and r3m
```
pip install git+https://github.com/openai/CLIP.git
```

```
pip install git+https://github.com/facebookresearch/r3m.git
```


## Training

### Conda env preparation:
```
conda env create -n rold python=3.10
conda activate rold
source rold_env.sh
```
