import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import PIL
from torchsummary import summary

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import configparser as ConfigParser

from utils import *
# Ipmport various network architectures
from networks import AGRadGalNet, VanillaLeNet, testNet, DNSteerableLeNet, DNSteerableAGRadGalNet
# Import various data classes
from datasets import FRDEEPF
from datasets import MiraBest_full, MBFRConfident, MBFRUncertain, MBHybrid
from datasets import MingoLoTSS, MLFR, MLFRTest
import torch.nn.functional as F
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-I','--iterations', default=100, type=int, required=True, help='Number of realisations of the Fisher Information Matrix')
parser.add_argument('-C','--config', default="myconfig.txt", type=str, required=True, help='Name of the input config file')
args, __ = parser.parse_known_args()
config_name = args['config']
n_iterations = int(args['iterations'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config      = ConfigParser.ConfigParser(allow_no_value=True)
config.read(f"configs/{config_name}")

workingdir = config['output']['directory']+'/'+config['data']['augment']

test_data_loader = utils.data.load(
    config,
    train=False,
    augmentation='config',
    data_loader=True
)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5],[0.5])])

print(f"Loading in {config['model']['base']}")
net = locals()[config['model']['base']](**config['model']).to(device)

quiet = config.getboolean('DEFAULT', 'quiet')
early_stopping = config.getboolean('training', 'early_stopping')

# Read / Create Folder for Data to be Saved
root = config['data']['directory']
os.makedirs(root, exist_ok=True)


path_supliment = config['data']['augment']+'/'
model = utils.utils.load_model(config, load_model='best', device=device, path_supliment=path_supliment)

train_loader = utils.data.load(
    config,
    train=False,
    augmentation='config',
    data_loader=True
)

utils.fisher.WeightTransfer(model, net)
del(model)
outputsize = config.getint('data', 'num_classes')
Fishers, Rank, FR = utils.fisher.CalcFIM(net, train_loader, n_iterations, outputsize, workingdir)


EV = []
nbins = 8
#normalise fisher
normalised_fishers = utils.fisher.normalise(Fishers.cpu(),outputsize*len(net.last_weights().weight[0]))
#Save Plots of the Eigenvalues, Rank and FR Norm to the relevant model
#----Calculate Effective Dimension Metrics----#
#Normalise the Fisher Matrix
ed = []
n_samples = [i for i in range(100,10000000,100)]
ed = utils.fisher.effective_dimension(net, normalised_fishers, outputsize*len(net.last_weights().weight[0]), n_samples, outputsize)
d = {"Samples": n_samples, "ED": np.array([x/outputsize*len(net.last_weights().weight[0]) for x in ed])}
pickle.dump(d, open(f"{workingdir}/effd.p", "wb"))
