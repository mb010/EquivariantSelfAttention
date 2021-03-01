import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import configparser
import numpy as np

class model(configfile):
    def __init__(self, configfile):
        # Read in the config file
        self.config = configparser.ConfigParser()
        self.config.read(configfile)

        self.network = utils.load_network(**config['network'])
        self.data = utils.load_data(**config['data']) # Should contain self.data.test and self.data.train (and maybe more?)
        #if config['data']['test_data'] != 'train':
        #    self.data = utils.load_data(config['data']['test_data'])
        #else:
        #    self.data = self.train_data

        if config['model']['trained']:
            self.model = utils.load_model(**config['model'])
            self.hyperparameters = self.config_dict(config['hypterparameters'])
        else:
            self.hyperparameters = utils.grid_search(**config['model'], self.data)
            self.model = utils.train(**config['model'], self.data)

        # Initialise various parameters
        #self.net_name = self.config['network']['name']
        #self.data_name = self.config['data']['name']
        #self.hypterparameters = self.config['parameters']
        #self.quiet = self.config['utils']['quiet']
        #self.loss = self.config['utils']['loss']
        #self.optimiser = self.config['utils']['optimiser'] # Hyperparameter?
        #self.config['save'] # If volume is too large)

        # Load objects
        #self.data = self.load_data()#**self.config['data']
        #self.net = self.get_net() # **self.config['network']
        #self.model = self.get_model() # self.config['network'],self.config['data'],self.config['hyperparameters']
