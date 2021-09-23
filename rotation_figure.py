import os
import subprocess
from tqdm import tqdm

import e2cnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import PIL

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import configparser as ConfigParser

import utils
# Ipmport various network architectures
from networks import AGRadGalNet, DNSteerableLeNet, DNSteerableAGRadGalNet, VanillaLeNet #e2cnn module only works in python3.7+
# Import various data classes
from datasets import FRDEEPF
from datasets import MiraBest_full, MBFR, MBFRConfident, MBFRUncertain, MBHybrid
from datasets import MingoLoTSS, MLFR, MLFRTest
from torchvision.datasets import MNIST


torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read in config file
args        = parse_args()
config_name = args['config']
config      = ConfigParser.ConfigParser(allow_no_value=True)
config.read(f"configs/{config_name}")
quiet = config.getboolean('DEFAULT', 'quiet')

net = locals()[config['model']['base']](**config['model']).to(device)
test_data = utils.data.load(
    config,
    train=False,
    augmentation='None',
    data_loader=False
)

save_name = "rotation_"
save_path = config["output"]["directory"]+"/"config["data"]["augment"]

for idx, (img, target) in enumerate(zip(test_data.data, test_data.targets)):
    mean, std, predictions = utils.figures.fr_rotation_test(
        model=model.to(device),
        data=torch.from_numpy(img.squeeze()[np.newaxis, np.newaxis, :, :]).to(device),
        target=target,
        idx=idx,
        device=device,
        save=save_path + "/" + save_name,
        show_images=False,
        figsize=(8,4),
        output='full'
    )
    np.save(f"{save_path}/rot_pred{idx}.npy", predictions)
    print(idx, mean, std)
