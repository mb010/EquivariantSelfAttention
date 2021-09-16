import os
import sys

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

args        = utils.parse_args()
config_name = args['config']
try:
    n_iterations = int(args['iterations'])
except:
    print("Please specify the number of iterations by the flag --iterations, set to 100 for default")
    n_iterations = 100;
config      = ConfigParser.ConfigParser(allow_no_value=True)
config.read(f"configs/{config_name}")

workingdir = config['output']['directory']

train_loader, valid_loader  = utils.data.load(
    config, 
    train=True, 
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

utils.fisher.WeightTransfer(model, net)
del(model)
Fishers, Rank, FR = utils.fisher.CalcFIM(net, train_loader, n_iterations)

print("Saving Fisher Realisations to a Pickle File...")
pickle.dump(Fishers, open(f"{workingdir}/fishers.p", "wb"))

EV =[]
nbins = 8
plt.subplot(111)
#normalise fisher
normalised_fishers = utils.fisher.normalise(Fishers.cpu())
#Save Plots of the Eigenvalues, Rank and FR Norm to the relevant model
for i in Fishers:
    EV = np.append(EV,torch.eig(i, eigenvectors=False,  out=None)[0][:,0].detach().numpy())
    
plt.hist(EV, bins=nbins, rwidth=0.8, color='r')
plt.ylabel("Total Counts")
plt.xlabel("Eigenvalue")
plt.title("Fisher Matrix Eigenspectrum")
plt.savefig(f"{workingdir}/Eigen.png")
plt.close()


plt.subplot(111)

plt.hist(Rank, bins=nbins, rwidth=0.8, color='g', density=True)
plt.ylabel("Normalised Counts")
plt.xlabel("Matrix Rank")
plt.title("Fisher Matrix Rank Distribution")
MeanRank = np.mean(Rank)
print(f"The Mean Rank obtained is {MeanRank} \n")
plt.savefig(f"{workingdir}/Rank.png")

plt.close()
plt.subplot(111)

FR2 = [np.abs(e) for e in FR]
plt.hist(FR2, bins=nbins, rwidth = 0.8 ,color='b', density=True)
plt.ylabel("Normalised Counts")
plt.xlabel("Fisher Rao Norm")
plt.title("Fisher Rao Norm Distribution")
plt.savefig(f"{workingdir}/FRNorm.png")

#----Calculate Effective Dimension Metrics----#
#Normalise the Fisher Matrix
ed = []
n_samples = [i for i in range(100,10000000,100)]
ed = effective_dimension(net,normalised_fishers, 2, n_samples, 12)
d = {"Samples": n_samples, "ED": np.array([x/12 for x in ed])}
pickle.dump(d, open(f"{workingdir}/effd.p", "wb"))
plt.plot(d['Samples'], d['ED'])