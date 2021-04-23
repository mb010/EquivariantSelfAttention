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
    augmentation='', 
    data_loader=True
)

"""
# -----------------------------------------------------------------------------
# Create data transformations
datamean = config.getfloat('data', 'datamean')
datastd = config.getfloat('data', 'datastd')
number_rotations = config.getint('data', 'number_rotations')
imsize = config.getint('data', 'imsize')
scaling_factor = config.getfloat('data', 'scaling')
angles = np.linspace(0, 359, config.getint('data', 'number_rotations'))
p_flip = 0.5 if config.getboolean('data','flip') else 0
augment = config.getboolean('data', 'augment')

# Create hard random (seeded) rotation:
class RotationTransform:
    """#Rotate by one of the given angles.
    """
    def __init__(self, angles, interpolation):
        self.angles = angles
        self.interpolation = interpolation

    def __call__(self, x):
        angle = np.random.choice(a=self.angles, size=1)[0]
        return transforms.functional.rotate(x, angle, resample=self.interpolation)

# Compose dict of transformations
transformations = {
    'none': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([datamean],[datastd])
    ]),
    'rotation and flipping': transforms.Compose([
        transforms.CenterCrop(imsize),
        transforms.RandomVerticalFlip(p=p_flip),
        RotationTransform(angles, interpolation=PIL.Image.BILINEAR),
        transforms.RandomAffine(
            degrees=0, # No uncontrolled rotation
            scale=(1-scaling_factor, 1+scaling_factor), 
            resample=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([datamean],[datastd])
    ]),
    'no rotation no flipping': transforms.Compose([
        transforms.CenterCrop(imsize),
        transforms.RandomVerticalFlip(p=p_flip),
        transforms.RandomAffine(
            degrees=0, # No uncontrolled rotation
            scale=(1-scaling_factor, 1+scaling_factor), 
            resample=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([datamean],[datastd])
    ]),
    'random rotation and flipping': transforms.Compose([
        transforms.CenterCrop(imsize),
        transforms.RandomVerticalFlip(p=p_flip),
        transforms.RandomAffine(
            degrees=360, # Random rotation
            scale=(1-scaling_factor, 1+scaling_factor), 
            resample=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([datamean],[datastd])
    ])
}

download = True
train = True
data_class = locals()[config['data']['dataset']]

#augmentations = [
#    "rotation and flipping",
#    "random rotation",
#    "restricted random rotation"
#]

if augment=='True' and p_flip==0.5:
    transform = transformations['rotation and flipping']
elif
else:
    transform = transformations['no rotation no flipping']

train_data = data_class(root=root, download=download, train=True, transform=transform)

# -----------------------------------------------------------------------------
# Cross Validation Parameters
def fetch_grid_search_param(name, config=config):
    raw_ = config['grid_search'][name]
    raw_list = raw_.split(',')
    do = bool(raw_list.pop(0))
    out = np.asarray(raw_list, dtype=np.float64)
    return do, out

# -----------------------------------------------------------------------------
# Get data parameters
batch_size = config.getint('training', 'batch_size')
validation_size = config.getfloat('training', 'validation_set_size')
dataset_size = len(train_data)

nval = int(validation_size*dataset_size)
indices = list(range(dataset_size))
np.random.shuffle(indices)

# -----------------------------------------------------------------------------
# Split Training Data for (Cross) Validation
train_indices, val_indices = indices[nval:], indices[:nval]
train_sampler = torch.utils.data.Subset(train_data, train_indices)
valid_sampler = torch.utils.data.Subset(train_data, val_indices)

train_loader = torch.utils.data.DataLoader(train_sampler, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_sampler, batch_size=batch_size, shuffle=True)
"""
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
    root_out_directory_addition=f'{config['data']['augment']}',
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