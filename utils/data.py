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
from networks import AGRadGalNet, DNSteerableLeNet, DNSteerableAGRadGalNet #e2cnn module only works in python3.7+
# Import various data classes
from datasets import FRDEEPF
from datasets import MiraBest_full, MBFRConfident, MBFRUncertain, MBHybrid
from datasets import MingoLoTSS, MLFR, MLFRTest

# Set seeds for reproduceability
#torch.manual_seed(42)
#np.random.seed(42)

# -----------------------------------------------------------------------------

def load(config, train=False, augmentation='config', angle=0, data_loader=False):
    """Load data using config and specific secondary parameters
    Args:
    config: Parsed config object
        Formated as examples in github.com/mb010/EquivariantAttention
    train: bool
        Train (True) or test set (False) set.
    augmentation: str
        Selection of augmentation style. 'none', 'random rotation', 
        or other for augmentation according to config file.
    angle: float
        Specific angle for augmentation='rotation only'
    data_loader: bool
        Output dataloader or not 
        (outputs tuple of train & validation data loaders if train=True)
    
    Outputs:
    Augmented pytorch data
    or
    Data loader for the augmented pytorch data (tuple if train=True)
    """
    # Read / Create Folder for Data to be Saved
    root = config['data']['directory']
    os.makedirs(root, exist_ok=True)

    # Create data transformations
    datamean = config.getfloat('data', 'datamean')
    datastd = config.getfloat('data', 'datastd')
    number_rotations = config.getint('data', 'number_rotations')
    imsize = config.getint('data', 'imsize')
    scaling_factor = config.getfloat('data', 'scaling')
    angles = np.linspace(0, 359, config.getint('data', 'number_rotations'))
    p_flip = 0.5 if config.getboolean('data','flip') else 0
    #augment = config.getboolean('data', 'augment')
    augment = config['data']['augment']
    if augment in ['None', 'random rotation', 'restricted random rotation']:
        augmentation=augment

    # Create hard random (seeded) rotation:
    class RotationTransform:
        """Rotate by one of the given angles."""
        def __init__(self, angles, interpolation):
            self.angles = angles
            self.interpolation = interpolation

        def __call__(self, x):
            angle = np.random.choice(a=self.angles, size=1)[0]
            return transforms.functional.rotate(x, angle, resample=self.interpolation)

    # Compose dict of transformations
    transformations = {
        'None': transforms.Compose([
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
        'random rotation':transforms.Compose([
            transforms.CenterCrop(imsize),
            transforms.RandomVerticalFlip(p=p_flip),
            transforms.RandomAffine(
                degrees=360,
                scale=(1-scaling_factor, 1+scaling_factor),
                resample=PIL.Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([datamean],[datastd])
        ]),
        'restricted random rotation':transforms.Compose([
            transforms.CenterCrop(imsize),
            transforms.RandomVerticalFlip(p=p_flip),
            transforms.RandomAffine(
                degrees=360/8/2,
                scale=(1-scaling_factor, 1+scaling_factor),
                resample=PIL.Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([datamean],[datastd])
        ])
    }
    
    download = True
    data_class = globals()[config['data']['dataset']]
    if augmentation=='None':
        transform = transformations[augmentation]
    elif augmentation=='random rotation':
        transform = transformations[augmentation]
    elif augmentation=="restricted random rotation":
        transform = transformations[augmentation]
    else:
        if augment=='True' and p_flip==0.5:
            transform = transformations['rotation and flipping']
        else:
            transform = transformations['no rotation no flipping']

    data = data_class(root=root, download=download, train=train, transform=transform)
    #sources = data.data
    #labels = np.asarray(data.targets)
    
    if data_loader:
        batch_size = config.getint('training', 'batch_size')
        if train:
            # -----------------------------------------------------------------------------
            validation_size = config.getfloat('training', 'validation_set_size')
            dataset_size = len(train_data)

            nval = int(validation_size*dataset_size)
            indices = list(range(dataset_size))
            np.random.shuffle(indices)
            # Split Training Data for Validation
            train_indices, val_indices = indices[nval:], indices[:nval]
            train_sampler = torch.utils.data.Subset(data, train_indices)
            valid_sampler = torch.utils.data.Subset(data, val_indices)

            train_loader = torch.utils.data.DataLoader(train_sampler, batch_size=batch_size, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(valid_sampler, batch_size=batch_size, shuffle=True)
            loader = (train_loader, valid_loader)
        else:
            loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
        return loader
    else:
        return data