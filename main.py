import os

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
from networks import AGRadGalNet
# Import various data classes
from datasets import FRDEEPF
from datasets import MiraBest_full, MBFRConfident, MBFRUncertain, MBHybrid
from datasets import MingoLoTSS, MLFR, MLFRTest
#import argparse

# Set seeds for reproduceability
torch.manual_seed(42)
np.random.seed(42)

# Get correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read in config file
args = utils.parse_args()
config_name = args['config']
#config_name = "configs/bowles2021mirabest.cfg"
config = ConfigParser.ConfigParser(allow_no_value=True)
config.read(f"configs/{config_name}")

# Load network architecture (with random weights)
model_name = config['model']['base']
print(f"Loading in {model_name}")
net = locals()[config['model']['base']](**config['model']).to(device)

"""#class Model:
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
"""

# Create data transformations
datamean = config.getfloat('data', 'datamean')
datastd = config.getfloat('data', 'datastd')
number_rotations = config.getint('data', 'number_rotations')
imsize = config.getint('data', 'imsize')
scaling_factor = config.getfloat('data', 'scaling')
angles = np.linspace(0, 359, config.getint('data', 'number_rotations'))
print(f"angles: {angles}")
p_flip = 0.5 if config.getboolean('data','flip') else 0

# Create hard random (seeded) rotation:
class RotationTransform:
    """Rotate by one of the given angles."""
    def __init__(self, angles, resample):
        self.angles = angles
        self.resample = resample

    def __call__(self, x):
        angle = np.random.choice(a=self.angles, size=1)[0]
        print(f'ANGLE CALL: {angle}')
        return transforms.functional.rotate(x, angle, resample=self.resample)


# Compose dict of transformations
transformations = {
    'none': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([datamean],[datastd])
    ]),
    'rotation and flipping': transforms.Compose([
        transforms.CenterCrop(imsize),
        transforms.RandomVerticalFlip(p=p_flip),
        RotationTransform(angles, resample=PIL.Image.BILINEAR),
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
    ])
}

# Read / Create Folder for Data to be Saved
root = config['data']['directory']
os.makedirs(root, exist_ok=True)
download = True
train = True
data_class = locals()[config['data']['dataset']]
data_set = data_class(root=root, download=download, train=train, transform=transformations['rotation and flipping'])

# Seperate Data into Subsets for Training, Validation, Testing.
train_data = data_class(root=root, download=download, train=True, transform=transformations['rotation and flipping'])
test_data = data_class(root=root, download=download, train=False, transform=transformations['rotation and flipping'])

#train_batched = torch.utils.data.DataLoader(traindata, batch_size=config.getint('training', 'batch_size'))

# Get data parameters
batch_size = config.getint('training', 'batch_size')
validation_size = config.getfloat('training', 'validation_set_size')
dataset_size = len(train_data)
nval = int(validation_size*dataset_size)
indices = list(range(dataset_size))
np.random.shuffle(indices)

train_indices, val_indices = indices[nval:], indices[:nval]

train_sampler = torch.utils.data.Subset(train_data, train_indices)
valid_sampler = torch.utils.data.Subset(train_data, val_indices)

train_loader = torch.utils.data.DataLoader(train_sampler, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_sampler, batch_size=batch_size, shuffle=True)

learning_rate = config.getfloat('training', 'learning_rate')
optimizers = {
    'SGD': optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9),
    'Adagrad': optim.Adagrad(net.parameters(), lr=learning_rate),
    'Adadelta': optim.Adadelta(net.parameters(), lr=learning_rate),
    'Adam': optim.Adam(net.parameters(), lr=learning_rate)
}

augmentation_loops = config.getint('data', 'number_rotations')
if config.getboolean('data', 'flip'):
    augmentation_loops = augmentation_loops*2
save_validation_updates = True
class_splitting_index = 1
loss_function = nn.CrossEntropyLoss()
optim_name = config['training']['optimizer']
optimizer = optimizers[optim_name]
training_results = {
    'train_loss': 0,
    'validation_loss': 0,
    'TP': 0,
    'FP': 0,
    'FN': 0,
    'TN': 0,
    'validation_update': False
}
df = pd.DataFrame(columns = list(training_results.keys()))
folder_name = config['output']['directory']
output_evaluation_path = config['output']['training_evaluation']
output_models_path = config['output']['model_file']
os.makedirs(folder_name, exist_ok=True)

# Variable selections
validation_loss_min = np.Inf
Epoch = config.getint('training', 'epochs')
for epoch_count in range(Epoch):

    # Model Training
    train_loss = 0.
    validation_loss = 0.
    confussion_matrix = np.zeros((2,2))
    net.train() #Set network to train mode.
    if 'binary_labels' in locals():
        del binary_labels
    if 'outputs' in locals():
        del outputs

    # Loop across random data augmentations
    for i in range(augmentation_loops):
        for batch_idx , (data, labels) in enumerate(train_loader): #Iterates through each batch.
            data = data.to(device)
            labels = labels.to(device)

            # Create binary labels to remove morphological subclassifications (for MiraBest)
            binary_labels = np.zeros(labels.size(), dtype=int)
            binary_labels = np.where(labels.cpu().numpy()<class_splitting_index, binary_labels, binary_labels+1)
            binary_labels = torch.from_numpy(binary_labels).to(device)

            pred = net.forward(data)
            optimizer.zero_grad()
            loss = loss_function(pred,binary_labels)
            loss.backward(retain_graph=True)
            optimizer.step()
            train_loss += (loss.item()*data.size(0))


    ### Model Validation ###
    net.eval()
    for epoch_valid in range(augmentation_loops):
        for batch_idx, (data, labels) in enumerate(valid_loader):
            data = data.to(device)
            labels = labels.to(device)

            # Create binary labels to remove morphological subclassifications
            binary_labels = np.zeros(labels.size(), dtype=int)
            binary_labels = np.where(labels.cpu().numpy()<class_splitting_index, binary_labels, binary_labels+1)
            binary_labels = torch.from_numpy(binary_labels).to(device)

            outputs = net.forward(data)
            loss = loss_function(outputs, binary_labels)
            validation_loss += (loss.item()*data.size(0))
            
            predictions = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            target_values = binary_labels.detach().cpu().numpy()
            for x, y in zip(predictions, target_values):
                confussion_matrix[x,y] += 1

    # Average losses (scaled according to validation dataset size)
    validation_loss = validation_loss/(len(valid_loader.dataset)*augmentation_loops)
    train_loss = train_loss/(len(train_loader.dataset)*augmentation_loops)
    training_results['train_loss'] = train_loss
    training_results['validation_loss'] = validation_loss
    #training_results['validation_confussion_matrix'] = confussion_matrix
    training_results['TP'] = confussion_matrix[0,0]
    training_results['FP'] = confussion_matrix[0,1]
    training_results['FN'] = confussion_matrix[1,0]
    training_results['TN'] = confussion_matrix[1,1]

    # Print
    print(f"Epoch:{epoch_count:3}\tTraining Loss: {training_results['train_loss']:8.6f}\t\tValidation Loss: {training_results['validation_loss']:8.6f}")

    # Save model if validation loss decreased
    if validation_loss <= validation_loss_min:
        print(f"\tValidation Loss Down: \t({validation_loss_min:8.6f}-->{validation_loss:8.6f}) ... Updating saved model.")
        training_results['validation_update'] = True
        if save_validation_updates:
            torch.save(net.state_dict(), f"{folder_name}/{epoch_count}.pt")
        else:
            torch.save(net.state_dict(), f"{folder_name}/{output_models_path}")
        validation_loss_min = validation_loss
    else:
        training_results['validation_update'] = False

    
    # Save training loss / validation loss for plotting
    df = df.append(training_results, ignore_index=True)


df.to_csv(f'{folder_name}/{output_evaluation_path}', index=False)
print(f"\nFinished training.\nMinimum Validation Loss: {validation_loss_min:8.6}\n")

# Save final model, no matter the loss
torch.save(net.state_dict(), f'{folder_name}/last.pt')

outputs = net.forward(data)
print(outputs)
print(np.argmax(outputs.detach().cpu().numpy(), axis=1))
print(binary_labels.cpu().numpy())

print(f"confussion_matrix:\n{confussion_matrix}")
#plt.imshow(confussion_matrix, cmap='Blues')
#plt.xlabel('True Label')
#plt.ylabel('Predicted Label')
#plt.xticks([0,1],['FRI','FRII'])
#plt.yticks([0,1],['FRI','FRII'])
#plt.show()
print(f"confussion_matrix.sum():\t{confussion_matrix.sum()}")
print(f"sum should be:\t{int(len(train_data)*0.2)*16*2}")