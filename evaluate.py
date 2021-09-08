import os

import e2cnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import PIL

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import configparser as ConfigParser

import utils
# Ipmport various network architectures
from networks import AGRadGalNet, DNSteerableLeNet, DNSteerableAGRadGalNet, VanillaLeNet #e2cnn module only works in python3.7+
# Import various data classes
from datasets import FRDEEPF
from datasets import MiraBest_full, MBFRConfident, MBFRUncertain, MBHybrid
from datasets import MingoLoTSS, MLFR, MLFRTest
from torchvision.datasets import MNIST

from sklearn.metrics import classification_report, roc_curve, auc

# Parse config
args        = utils.parse_args()
config_name = args['config']
config      = ConfigParser.ConfigParser(allow_no_value=True)
config.read(f"configs/{config_name}")
data_config = ConfigParser.ConfigParser(allow_no_value=True)

# Set seeds for reproduceability
torch.manual_seed(42)
np.random.seed(42)
# Get correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data sets to iterate over
data_configs = [
    "5kernel_bowles2021_mirabest_RandAug.cfg", # Mirabest Dataset - MBFR
    "5kernel_bowles2021_mingolotss_RandAug.cfg" # Mingo Dataset - MLFR
]

# Evaluation augmentations to iterate over
augmentations = [
    "random rotation",
    #"restricted random rotation"
]


csv_path = config['output']['directory'] +'/'+ config['data']['augment'] +'/'+ config['output']['training_evaluation']
df = pd.read_csv(csv_path)
best = df.iloc[list(df['validation_update'])].iloc[-1]

# Extract models kernel size
if config.has_option('model', 'kernel_size'):
    kernel_size = config.getint('model', 'kernel_size')
elif "LeNet" in config['model']['base']:
    kernel_size = 5
else:
    kernel_size = 3

net = locals()[config['model']['base']](**config['model']).to(device)

if config['data']['dataset'] == 'MNIST':
    data_configs = [config_name]
for d_cfg in data_configs:
    for augmentation in augmentations:
        path_supliment = config['data']['augment']+'/'
        model = utils.utils.load_model(config, load_model='best', device=device, path_supliment=path_supliment)
        data_config = ConfigParser.ConfigParser(allow_no_value=True)
        data_config.read('configs/'+d_cfg)
        data  = utils.data.load(
            data_config,
            train=False,
            augmentation=augmentation,
            data_loader=True
        )

        y_pred, y_labels = utils.evaluation.predict(
            model,
            data,
            augmentation_loops=100,
            raw_predictions=True,
            device=device,
            verbose=True
        )

        utils.evaluation.save_evaluation(
            y_pred,
            y_labels,
            model_name=config['model']['base'],
            kernel_size=kernel_size,
            train_data=config['data']['dataset'],
            train_augmentation=config['data']['augment'],
            test_data=data_config['data']['dataset'],
            test_augmentation=augmentation,
            epoch=int(best.name),
            best=True,
            raw=False,
            PATH='/share/nas/mbowles/EquivariantSelfAttention/' + config['output']['directory'] +'/'+ config['data']['augment']+'/'
        )
