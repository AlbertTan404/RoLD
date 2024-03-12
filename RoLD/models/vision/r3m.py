import os
import copy
import gdown
import omegaconf
import hydra
from PIL import Image

import torch
import torchvision.transforms as T


def load_r3m(modelid):
    VALID_ARGS = ["_target_", "device", "lr", "hidden_dim", "size", "l2weight", "l1weight", "langweight", "tcnweight", "l2dist", "bs"]
    def remove_language_head(state_dict):
        keys = state_dict.keys()
        ## Hardcodes to remove the language head
        ## Assumes downstream use is as visual representation
        for key in list(keys):
            if ("lang_enc" in key) or ("lang_rew" in key):
                del state_dict[key]
        return state_dict

    def cleanup_config(cfg):
        config = copy.deepcopy(cfg)
        keys = config.agent.keys()
        for key in list(keys):
            if key not in VALID_ARGS:
                del config.agent[key]
        config.agent["_target_"] = "r3m.R3M"
        config["device"] = 'cpu'
        
        ## Hardcodes to remove the language head
        ## Assumes downstream use is as visual representation
        config.agent["langweight"] = 0
        return config.agent

    home = os.path.join(os.path.expanduser("~"), ".r3m")
    if modelid == "resnet50":
        foldername = "r3m_50"
        modelurl = 'https://drive.google.com/uc?id=1Xu0ssuG0N1zjZS54wmWzJ7-nb0-7XzbA'
        configurl = 'https://drive.google.com/uc?id=10jY2VxrrhfOdNPmsFdES568hjjIoBJx8'
    elif modelid == "resnet34":
        foldername = "r3m_34"
        modelurl = 'https://drive.google.com/uc?id=15bXD3QRhspIRacOKyWPw5y2HpoWUCEnE'
        configurl = 'https://drive.google.com/uc?id=1RY0NS-Tl4G7M1Ik_lOym0b5VIBxX9dqW'
    elif modelid == "resnet18":
        foldername = "r3m_18"
        modelurl = 'https://drive.google.com/uc?id=1A1ic-p4KtYlKXdXHcV2QV0cUzI4kn0u-'
        configurl = 'https://drive.google.com/uc?id=1nitbHQ-GRorxc7vMUiEHjHWP5N11Jvc6'
    else:
        raise NameError('Invalid Model ID')

    if not os.path.exists(os.path.join(home, foldername)):
        os.makedirs(os.path.join(home, foldername))
    modelpath = os.path.join(home, foldername, "model.pt")
    configpath = os.path.join(home, foldername, "config.yaml")
    if not os.path.exists(modelpath):
        gdown.download(modelurl, modelpath, quiet=False)
        gdown.download(configurl, configpath, quiet=False)
        
    modelcfg = omegaconf.OmegaConf.load(configpath)
    cleancfg = cleanup_config(modelcfg)
    rep = hydra.utils.instantiate(cleancfg)
    r3m_state_dict = remove_language_head(torch.load(modelpath, map_location=torch.device('cpu'))['r3m'])
    filtered_state_dict = {}
    for key, value in r3m_state_dict.items():
        if key.startswith("module"):
            new_key = key.replace("module.", "")
            filtered_state_dict[new_key] = value
    rep.load_state_dict(filtered_state_dict)
    return rep


class R3M(torch.nn.Module):
    def __init__(
        self,
        model_id: str
    ):
        super().__init__()
        r3m = load_r3m(model_id)

        self.model = r3m.convnet
        self.normlayer = r3m.normlayer
        self.feature_size = r3m.outdim

        self.preprocess = T.Compose([
            T.ToTensor(),
            self.normlayer
        ])
    
    def raw_preprocess(self, image: Image):
        # depreciated
        shorter_edge = min(image.size)
        process = T.Compose([
            T.CenterCrop(shorter_edge),
            T.Resize(224),
            T.ToTensor(),
            self.normlayer
        ])
        return process(image)
    
    def forward(self, images):
        return self.model(images)
