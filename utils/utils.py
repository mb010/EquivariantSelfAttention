import argparse
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


def parse_args():
    """
        Parse the command line arguments
        """
    parser = argparse.ArgumentParser()
    parser.add_argument('-C','--config', default="myconfig.txt", required=True, help='Name of the input config file')
    
    args, __ = parser.parse_known_args()
    
    return vars(args)

def load_network(config, device):
    model = globals()[config['model']['base']](**config['model']).to(device)
    return model

def load_model(config, load_model='best', path_supliment='', device=torch.device('cpu')):
    """Load in a model of choice.
    config:
    load_model: str
        Condition for model selection
    path_supliment: str
        Between output and directory name, only needed 
        for sub-grid search selection in this framework.
    device: obj
        torch device specification for initial loading of model weights.
    """
    # Load training csv
    csv_path = config['output']['directory'] +'/'+ path_supliment + config['output']['training_evaluation']
    df = pd.read_csv(csv_path)
    
    if load_model=='best':
        # Extract best model
        best = df.iloc[list(df['validation_update'])].iloc[-1]
        best_epoch = int(best.name)
        MODEL_PATH = config['output']['directory'] +'/'+ path_supliment + str(best_epoch) + '.pt'
    
    model = globals()[config['model']['base']](**config['model']).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    return model