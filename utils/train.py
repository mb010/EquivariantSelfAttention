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

def train(net, 
          device, 
          config,
          train_loader,
          valid_loader,
          optimizer,
          root_out_directory_addition='',
          scheduler = None,
          save_validation_updates=True,
          class_splitting_index=1,
          loss_function = nn.CrossEntropyLoss(),
          output_model=True,
          early_stopping=True,
          output_best_validation=False,
          stop_after_epochs_without_update=2000
         ):
    """Trains a network with a given config file on 
    the training and validation sets as provided.
    """
    # -----------------------------------------------------------------------------
    # Initialise Seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Read in values
    quiet          = config.getboolean('DEFAULT', 'quiet')
    early_stopping = config.getboolean('training', 'early_stopping')
    
    # Initialise training / validation saving objects
    training_results = {
        'train_loss': 0,
        'validation_loss': 0,
        'TP': 0,
        'FP': 0,
        'FN': 0,
        'TN': 0,
        'validation_update': False
    }

    # Where to save output
    folder_name = config['output']['directory']
    folder_name += root_out_directory_addition
    output_evaluation_path = config['output']['training_evaluation']
    output_models_path = config['output']['model_file']
    os.makedirs(folder_name, exist_ok=True)
    print(f"Folder to be saved to: {folder_name}")
    print(f"Is PATH: {os.path.isdir(folder_name)}")
    
    # Initialise data frame and CSV file
    df = pd.DataFrame(columns = list(training_results.keys()))
    df.to_csv(f'{folder_name}/{output_evaluation_path}', index=False)

    # Training parameters
    augmentation_loops = config.getint('data', 'number_rotations')
    if config.getboolean('data', 'flip'):
        augmentation_loops = augmentation_loops*2
    # Potentially optimise early stopping augment validation set size?
    #if config['data']['augment'] == "random rotation" & "DN" in config['model']['base']:
    #    augmentation_loops = 10*augmentaion_loops
    
    # -----------------------------------------------------------------------------
    # Training Loop
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

        # Loop across data augmentations
        for i in range(augmentation_loops):
            for batch_idx , (data, labels) in enumerate(train_loader): #Iterates through each batch.
                data = data.to(device)
                labels = labels.to(device)

                # Create binary labels to remove morphological subclassifications (for MiraBest) ### IS THIS STILL NECESSARY?
                binary_labels = np.zeros(labels.size(), dtype=int)
                binary_labels = np.where(labels.cpu().numpy()<class_splitting_index, binary_labels, binary_labels+1)
                binary_labels = torch.from_numpy(binary_labels).to(device)
                
                # Loss & backpropagation
                pred = net.forward(data)
                optimizer.zero_grad()
                loss = loss_function(pred,binary_labels)
                loss.backward(retain_graph=True)
                if scheduler == None:
                    optimizer.step()
                train_loss += (loss.item()*data.size(0))
            if scheduler != None:
                scheduler.step(train_loss)


        # Model Validation
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
        training_results['TP'] = confussion_matrix[0,0]
        training_results['FP'] = confussion_matrix[0,1]
        training_results['FN'] = confussion_matrix[1,0]
        training_results['TN'] = confussion_matrix[1,1]

        if not quiet: 
            print(f"Epoch:{epoch_count:3}\tTraining Loss: {training_results['train_loss']:8.6f}"
                  f"\t\tValidation Loss: {training_results['validation_loss']:8.6f}")

        # Save model if validation loss decreased
        if early_stopping and validation_loss <= validation_loss_min:
            best_confussion_matrix = confussion_matrix
            if not quiet:
                print(f"\tValidation Loss Down: \t({validation_loss_min:8.6f}-->{validation_loss:8.6f}) ... Updating saved model.")
            training_results['validation_update'] = True
            if save_validation_updates:
                torch.save(net.state_dict(), f"{folder_name}/{epoch_count}.pt")
            else:
                torch.save(net.state_dict(), f"{folder_name}/{output_models_path}")
            validation_loss_min = validation_loss
        else:
            training_results['validation_update'] = False

        # Save current losses / evaluations to the data frame
        df = df.append(training_results, ignore_index=True)
        
        if training_results['validation_update']:
            epochs_without_update=0
            # Appending data fram to csv
            df.to_csv(f"{folder_name}/{output_evaluation_path}", mode='a', index=False, header=False)
            df = pd.DataFrame(columns = list(training_results.keys()))
        
        # Implement break if no update has been found for given number of epochs (default 1000)
        epochs_without_update += 1
        if epochs_without_update >= stop_after_epochs_without_update:
            break
    
    # -----------------------------------------------------------------------------
    df.to_csv(f"{folder_name}/{output_evaluation_path}", mode='a', index=False, header=False)
    print(f"\nFinished training.\nMinimum Validation Loss: {validation_loss_min:8.6}\n")
    
    # Save final model
    torch.save(net.state_dict(), f'{folder_name}/last.pt')
    
    output = []
    if output_model:
        output.append(net)
    if output_best_validation:
        output.append(best_confussion_matrix)
        output.append(validation_loss_min)
    else:
        if early_stopping:
            print(f"best_confussion_matrix:\n{best_confussion_matrix}")
    return output

