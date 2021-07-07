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
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
        Specific model or a epoch number that has been saved.
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
    else:
        MODEL_PATH = config['output']['directory'] +'/'+ path_supliment + str(load_model) + '.pt'
    
    model = globals()[config['model']['base']](**config['model']).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    return model

def plot_3(
    a, b, c, 
    cmaps=['coolwarm'],
    titles=["FRI Mean", "FRII Mean", "Difference"],
    cbars_bool=[True],
    figsize=(16,9),
    vmin = ['adaptive'],
    vmax = ['adaptive'],
    factors = [1, -1, 1],
    save='',
    contour=''
):
    plt_contour=False
    if contour!='':
        plt_contour=True
    if len(cmaps)==1:
        cmaps *= 3
    if len(cbars_bool)==1:
        cbars_bool *= 3
    if len(vmin)==1:
        vmin *= 3
    if len(factors)==1:
        factors *= 3
    if len(vmax)==1:
        vmax *= 3
    
    if len(a.shape)==3:
        factors[0]=[1]
        a /=a.max()
    if len(b.shape)==3:
        b /=b.max()
    if len(c.shape)==3:
        c = c-c.min(axis=2, keepdims=True)
        c = c/c.max(axis=2, keepdims=True)
    
    
    layout = """
        abc
    """

    def lim(value):
        if value == "adaptive":
            func = lambda arr : np.max(np.abs(arr))
        else:
            func = lambda arr : value
        return func
    
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    ax_dict = fig.subplot_mosaic(layout)
    im_a = ax_dict['a'].imshow(factors[0]*a, cmap=cmaps[0], vmin=-lim(vmin[0])(a), vmax=lim(vmax[0])(a), origin='lower')
    im_b = ax_dict['b'].imshow(factors[1]*b, cmap=cmaps[1], vmin=-lim(vmin[1])(b), vmax=lim(vmax[1])(b), origin='lower')
    im_c = ax_dict['c'].imshow(factors[2]*c, cmap=cmaps[2], vmin=-lim(vmin[2])(c), vmax=lim(vmax[2])(c), origin='lower')
    if plt_contour:
        im_contour = ax_dict['c'].contour(contour, 0, levels=[0.])
    
    ax_dict['a'].set_xticks([]); ax_dict['a'].set_yticks([])
    ax_dict['b'].set_xticks([]); ax_dict['b'].set_yticks([])
    ax_dict['c'].set_xticks([]); ax_dict['c'].set_yticks([])
    
    ax_dict['a'].set_title(f"{titles[0]}", fontsize=14)
    ax_dict['b'].set_title(f"{titles[1]}", fontsize=14)
    ax_dict['c'].set_title(f"{titles[2]}", fontsize=14)
    
    if cbars_bool[0]:
        divider_a = make_axes_locatable(ax_dict['a'])
        cax_a = divider_a.append_axes('bottom', size='5%', pad=0.05)
        fig.colorbar(im_a, cax_a, orientation='horizontal')
    if cbars_bool[1]:
        divider_b = make_axes_locatable(ax_dict['b'])
        cax_b = divider_b.append_axes('bottom', size='5%', pad=0.05)
        fig.colorbar(im_b, cax_b, orientation='horizontal')
    if cbars_bool[2]:
        divider_c = make_axes_locatable(ax_dict['c'])
        cax_c = divider_c.append_axes('bottom', size='5%', pad=0.05)
        fig.colorbar(im_c, cax_c, orientation='horizontal')
    
    
    fig.show()
    if save != '':
        plt.savefig(save)
    
def build_mask(s, margin=2, dtype=torch.float32):
    mask = torch.zeros(1, 1, s, s, dtype=dtype)
    c = (s-1) / 2
    t = (c - margin/100.*c)**2
    sig = 2.
    for x in range(s):
        for y in range(s):
            r = (x - c) ** 2 + (y - c) ** 2
            if r > t:
                mask[..., x, y] = np.exp((t - r)/sig**2)
            else:
                mask[..., x, y] = 1.
    return mask