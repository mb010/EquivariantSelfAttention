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

from sklearn.metrics import classification_report, roc_curve, auc

# Set seeds for reproduceability
torch.manual_seed(42)
np.random.seed(42)

# Get correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

configs = [
    #"scaife2021mirabest.cfg", # Fully Evaluated
    #"scaife2021mirabest-RandAug.cfg", # Fully Evaluated
    #"scaife2021mirabest-RestrictedAug.cfg", # Fully Evaluated
    #"scaife2021mingo.cfg", # Fully Evaluated
    #"scaife2021mingo-RandAug.cfg", # Fully Evaluated
    #"scaife2021mingo-RestrictedAug.cfg", # Fully Evaluated
    
    #"bowles2021mirabest.cfg", # Fully Evaluated
    #"bowles2021mirabest-RandAug.cfg", # Fully Evaluated
    #"bowles2021mirabest-RestrictedAug.cfg", # Fully Evaluated
    #"bowles2021mingo.cfg", # Fully Evaluated
    #"bowles2021mingo-RandAug.cfg", # Fully Evaluated
    #"bowles2021mingo-RestrictedAug.cfg", # Fully Evaluated
    
    #"e2attentionmirabest.cfg", # Fully Evaluated
    #"e2attentionmirabest-RandAug.cfg", # Fully Evaluated
    #"e2attentionmirabest-RestrictedAug.cfg", # Fully Evaluated
    #"e2attentionmingo.cfg", # Fully Evaluated
    #"e2attentionmingo-RandAug.cfg", # Fully Evaluated
    #"e2attentionmingo-RestrictedAug.cfg", # Fully Evaluated
    
    #"5kernel_e2attentionmirabest.cfg",
    #"5kernel_e2attentionmirabest-RandAug.cfg", # Fully Evaluated
    #"5kernel_e2attentionmirabest-RestrictedAug.cfg",
    #"5kernel_e2attentionmingo.cfg",
    #"5kernel_e2attentionmingo-RandAug.cfg", # Fully Evaluated
    #"5kernel_e2attentionmingo-RestrictedAug.cfg",
    
    #"7kernel_e2attentionmirabest.cfg",
    #"7kernel_e2attentionmirabest-RandAug.cfg", # Fully Evaluated
    #"7kernel_e2attentionmirabest-RestrictedAug.cfg",
    #"7kernel_e2attentionmingo.cfg",
    #"7kernel_e2attentionmingo-RandAug.cfg", # Fully Evaluated
    #"7kernel_e2attentionmingo-RestrictedAug.cfg",
    
    "C4_attention_mirabest.cfg", 
    "C8_attention_mirabest.cfg", 
    "C16_attention_mirabest.cfg", 
    "D4_attention_mirabest.cfg", 
    "D8_attention_mirabest.cfg", 
    "D16_attention_mirabest.cfg", 
]

data_configs = [
    "e2attentionmirabest.cfg", # Mirabest Dataset - MBFR
    "e2attentionmingo.cfg" # Mingo Dataset - MLFR
]
augmentations = [
    #"rotation and flipping",
    "random rotation",
    #"restricted random rotation"
]

for cfg in configs:
    print(cfg)
    config = ConfigParser.ConfigParser(allow_no_value=True)
    data_config = ConfigParser.ConfigParser(allow_no_value=True)
    config.read('configs/'+cfg)
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
    
    for d_cfg in data_configs:
        for augmentation in augmentations:
            path_supliment = config['data']['augment']+'/'
            model = utils.utils.load_model(config, load_model='best', device=device, path_supliment=path_supliment)
            data_config.read('configs/'+d_cfg)
            print(f"Evaluating {cfg}: {config['output']['directory']}/{config['data']['augment']}\t{data_config['data']['dataset']}\t{augmentation}")
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
                PATH='/share/nas/mbowles/EquivariantSelfAttention/models/e2attention/mirabest/fisher'
            )