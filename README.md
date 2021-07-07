# EquivariantSelfAttention
An implementation of Equivariant e2 convolutional kernals into a convolutional self attention network, applied to radio astronomy data.


This work extends and builds on previous work:
 - [FR Classification using e2cnn](https://arxiv.org/pdf/2102.08252.pdf), whose code is [here](https://github.com/as595/E2CNNRadGal).
 - [Attention for FR Classificaion](https://arxiv.org/abs/2012.01248), whose code is [here](https://github.com/mb010/AstroAttention).
This work makes extensive use of the `e2cnn` package: [e2cnn](https://github.com/QUVA-Lab/e2cnn).

### Imports for General Use
```
import os
import torch
import e2cnn
import pandas as pd
import numpy as np
import configparser as ConfigParser

import utils
```

### Loading a network:
To load a network import the network files, read in a config file, and load the network with the config file parameters:
```
from networks import AGRadGalNet, VanillaLeNet, testNet, DNSteerableLeNet, DNSteerableAGRadGalNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config      = ConfigParser.ConfigParser(allow_no_value=True)
config_path = "configs/CONFIG_NAME"
config.read(config_path)

net = locals()[config['model']['base']](**config['model']).to(device)
```

Alternatively, if you know which architecture you want to use, only import that and load in the config parameters (i.e. without using `locals()`):
```
from networks import DNSteerableAGRadGalNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config      = ConfigParser.ConfigParser(allow_no_value=True)
config_path = "configs/CONFIG_NAME"
config.read(config_path)

net = DNSteerableAGRadGalNet(**config['model']).to(device)
```

### Loading a model: 
To load the best performing model (according to early stopping) from a given model training session:
```
path_supliment = config['data']['augment']+'/'
model = utils.utils.load_model(config, load_model='best', device=device, path_supliment=path_supliment)
```

### Testing a model: 
Follow the example of [evaluations.ipynb](./evaluations.ipynb) for model evaluation.

### Defining a model: 
To define your own model simply create a new config which follows the examples provided in [configs](./configs).

### Training a model: 
To train a model use the [train.py](./train.py) using your config file:
```
python3.8 train.py --config YOUR_CONFIG_NAME.cfg >& logs/YOUR_CONFIG_NAME.log
```
