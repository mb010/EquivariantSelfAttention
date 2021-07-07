import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchsummary import summary

import PIL

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import configparser as ConfigParser

import utils
# Ipmport various network architectures
from networks import AGRadGalNet, VanillaLeNet, testNet, DNSteerableLeNet, DNSteerableAGRadGalNet #e2cnn module only works in python3.7+
# Import various data classes
from datasets import FRDEEPF
from datasets import MiraBest_full, MBFRConfident, MBFRUncertain, MBHybrid
from datasets import MingoLoTSS, MLFR, MLFRTest


# -----------------------------------------------------------------------------
# Set seeds for reproduceability
torch.manual_seed(42)
np.random.seed(42)

# Get correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read in config file
args        = utils.parse_args()
config_name = args['config']
config      = ConfigParser.ConfigParser(allow_no_value=True)
config.read(f"configs/{config_name}")

quiet = config.getboolean('DEFAULT', 'quiet')
early_stopping = config.getboolean('training', 'early_stopping')

# Read / Create Folder for Data to be Saved
root = config['data']['directory']
os.makedirs(root, exist_ok=True)

# -----------------------------------------------------------------------------
# Load network architecture (with random seeded weights)
print(f"Loading in {config['model']['base']}")
net = locals()[config['model']['base']](**config['model']).to(device)

if not quiet:
    if 'DN' not in config['model']['base']:
        summary(net, (1, 150, 150))
    print(device)
    if device == torch.device('cuda'):
        print(torch.cuda.get_device_name(device=device))

train_loader, valid_loader  = utils.data.load(
    config, 
    train=True, 
    augmentation='config', 
    data_loader=True
)

# -----------------------------------------------------------------------------
# Cross Validation Parameters
def fetch_grid_search_param(name, config=config):
    raw_ = config['grid_search'][name]
    raw_list = raw_.split(',')
    do = bool(raw_list.pop(0))
    out = np.asarray(raw_list, dtype=np.float64)
    return do, out

# -----------------------------------------------------------------------------
# Extract learning values
learning_rate = config.getfloat('training', 'learning_rate')
do, lr_scaling = fetch_grid_search_param(name='learning_rate', config=config)
learning_rate *= lr_scaling
#print(learning_rate)

optim_name = config['training']['optimizer']

# -----------------------------------------------------------------------------
# Train
weight_decay = config.getfloat('training', 'weight_decay')
root_out_directory_addition = '/'+config['data']['augment']
lr = config.getfloat('final_parameters', 'learning_rate')
optimizers = {
    'SGD': optim.SGD(net.parameters(), lr=lr, momentum=0.9),
    'Adagrad': optim.Adagrad(net.parameters(), lr=lr),
    'Adadelta': optim.Adadelta(net.parameters(), lr=lr),
    'Adam': optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    }
optimizer  = optimizers[optim_name]
model, conf_mat, validation_min = utils.train(
    net,
    device,
    config,
    train_loader,
    valid_loader,
    optimizer=optimizer,
    root_out_directory_addition=root_out_directory_addition,
    scheduler = None,
    save_validation_updates=True,
    class_splitting_index=1,
    loss_function=nn.CrossEntropyLoss(),
    output_model=True,
    early_stopping=True,
    output_best_validation=True        
)
print(f"""Confusion Matrix: {conf_mat}
Learning Rate: {lr}
Minimal Validation Loss: {validation_min}
""")