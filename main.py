import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import configparser as ConfigParser
import numpy as np

import utils
import networks

# Set seeds for reproduceability
torch.manual_seed(42)
np.radnom.seed(42)


class Model:
    def __init__(self, configfile):
        # Read in the config file
        self.config_name = configfile
        self.config = ConfigParser.SafeConfigParser(allow_no_value=True)
        self.config.read(self.config_name)

        self.n_classes = self.config['data']['num_classes']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = utils.net.load(
            device=self.device,
            **self.config['model'])
        self.data = utils.data.load(
            device=self.device,
            rotations=self.config.getint('model', 'number_rotations'),
            **self.config['data'])
        self.train = utils.train.load(self.config_name)

        self.model_trained = self.config.getboolean('grid_search', 'done')


    def model_trained(self):
        if self.config.getboolean('grid_search', 'done'):
            print('Training not required')
            #self.model = utils.load_model(**config['model'])
            #self.hyperparameters = self.config(**config['final_parameters'])
        else:
            print('Training required')
            #self.hyperparameters = utils.grid_search(self.data, **config['model'])
            #self.model = utils.train(**config['model'], self.data)
