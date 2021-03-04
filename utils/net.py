import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import configparser as ConfigParser
import numpy as np


def hello_world():
    print('hello world')

def load(
    device,
    base,
    optimiser,
    early_stopping,
    number_rotations,
    attention_module,
    attention_gates,
    attention_normalisation,
    attention_aggregation,
    quiet):
    print('reached')
