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

# Import various network architectures
from networks import AGRadGalNet, DNSteerableLeNet, DNSteerableAGRadGalNet #e2cnn module only works in python3.7+
# Import various data classes
from datasets import FRDEEPF
from datasets import MiraBest_full, MBFRConfident, MBFRUncertain, MBHybrid
from datasets import MingoLoTSS, MLFR, MLFRTest

# sklearn
from sklearn.metrics import classification_report, roc_curve, auc

# Set seeds for reproduceability
torch.manual_seed(42)
np.random.seed(42)

def predict(model, 
            data, 
            augmentation_loops=1, 
            raw_predictions=True,
            device = torch.device('cpu')
           ):
    """Predict on multiple passes of input images
    Args:
    model: object
        pytorch model loaded with trained weights.
    data: object
        pytorch data loader data set to be iterated over during evaluation.
    augmentation_loops: int
        Number of passes over whole data sample.
    raw_predictions: boolean
        Condition to output list of bits (False) or 
        tuples of predictions with dimensions for each pass (True).
    
    Outputs:
    y_pred: np.array
        Models output for given input source
        Size: (augmentaiton_loops, total data size, no. classes) if raw_predictions
              (augmentation_loops * total data size) if not raw_predictions
    y_labels: np.array
        Data labels corresponding to predictions
        Size: (augmentation_loops, total data size) if raw_predictions
              (augmentation_loops * total data size) if not raw_predictions
        
    """
    model = model.to(device)
    for loop in range(augmentation_loops):
        for idx, batch in enumerate(data):
            pred = model(batch[0].to(device)).detach().cpu().numpy()
            labels = batch[1].detach().cpu().numpy()
            if idx>0:
                batch_pred = np.append(batch_pred, pred, axis=0)
                batch_labels = np.append(batch_labels, labels, axis=0)
            else:
                batch_pred = pred
                batch_labels = labels
        if loop == 0:
            y_pred = np.expand_dims(batch_pred, axis=0)
            y_labels = np.expand_dims(batch_labels, axis=0)
        else:
            y_pred = np.append(y_pred, np.expand_dims(batch_pred, axis=0), axis=0)
            y_labels = np.append(y_labels, np.expand_dims(batch_labels, axis=0), axis=0)
    
    if not raw_predictions:
        y_pred = list(y_pred.argmax(axis=2).flatten())
        y_labels = list(y_labels.flatten())
    return y_pred, y_labels

def save_evaluation(y_pred, 
                    y_labels, 
                    model_name='', 
                    test_data='', 
                    test_augmentation='', 
                    train_data='', 
                    train_augmentation='', 
                    epoch=np.nan,
                    best=False,
                    raw=False, 
                    PATH='./',
                    print_report = False
                   ):
    """Calculates evaluation metrics according 
    to provided predictions and data labels 
    and saves them to a csv.
    """
    if raw:
        # Cant be this naive. 3dim arrays wont save like this.
        #np.savetxt(PATH+'raw_predictions.txt', y_pred)
        #np.savetxt(PATH+'raw_labels.txt', y_labels)
        pass
    else:
        shape = (y_pred.shape[0]*y_pred.shape[1], y_pred.shape[2])
        pred = F.softmax(
            torch.from_numpy(y_pred.reshape(shape)),
            dim=1
        )
        fpr, tpr, thresholds = roc_curve(y_labels.flatten(), pred[:,1].numpy())
        AUC = auc(fpr, tpr)
        report = classification_report(
            y_labels.flatten(),
            y_pred.argmax(axis=2).flatten(),
            output_dict=True
        )
        if print_report:
            classification_report(
                y_labels.flatten(),
                y_pred.argmax(axis=2).flatten(),
                output_dict=False
            )
        evaluation = {
            'model': [model_name],
            'train_data': [train_data],
            'train_augmentation': [train_augmentation],
            'test_data': [test_data],
            'test_augmentation': [test_augmentation],
            'epoch': [epoch],
            'best': [best],
            '0 precision': [report['0']['precision']],
            '0 recall': [report['0']['recall']],
            '0 f1-score': [report['0']['f1-score']],
            '0 support': [report['0']['support']],
            '1 precision': [report['1']['precision']],
            '1 recall': [report['1']['recall']],
            '1 f1-score': [report['1']['f1-score']],
            '1 support': [report['1']['support']],
            'auc': [AUC]
        }
        df = pd.DataFrame.from_dict(evaluation)
        if os.path.isfile(PATH+'full_evaluations.csv'):
            df.to_csv(PATH+"full_evaluations.csv", mode='a', index=False, header=False)
        else:
            df.to_csv(PATH+"full_evaluations.csv", mode='w', index=False, header=True)
            
def training_plot(config, plot='', path_supliment='', marker='o', alpha=0.7, lr_modifier=1, fontsize=20, ylim=None, mean=False, window_size=1):
    if plot == '':
        plot = ['validation_loss', 'training_loss', 'accuracy']
    
    csv_path = config['output']['directory'] +'/'+ path_supliment + config['output']['training_evaluation']
    df = pd.read_csv(csv_path)
    epochs = np.asarray(range(len(df)))
    accuracy = (df.TP + df.TN)/(df.TP+df.TN + df.FP+df.FN)
    
    plt.rcParams.update({'font.size': fontsize})
    fig, ax = plt.subplots(figsize=(16,9))
    ax.grid()
    if mean:
        if 'training_loss' in plot:
            mean = df['train_loss'].rolling(window_size).mean()
            std = df['train_loss'].rolling(window_size).std()
            ax.plot(epochs, mean, label='Train Loss')
            ax.fill_between(epochs, mean-std, mean+std, alpha=0.3)
            #ax.scatter(epochs, df['train_loss'], label='train loss', marker=marker, alpha=alpha)
        if 'validation_loss' in plot:
            mean = df['validation_loss'].rolling(window_size).mean()
            std = df['validation_loss'].rolling(window_size).std()
            validation_min = df['validation_loss'][df['validation_loss'].idxmin]
            ax.plot(epochs, mean, label=f'Validation Loss ({validation_min:.3f})')
            ax.fill_between(epochs, mean-std, mean+std, alpha=0.3)
            #ax.scatter(epochs, df['validation_loss'], label='val. loss', marker=marker, alpha=alpha)
        if 'accuracy' in plot:
            mean = accuracy.rolling(window_size).mean()
            std = accuracy.rolling(window_size).std()
            model_acc = accuracy[df['validation_loss'].idxmin]
            ax.plot(epochs, mean, label=f'Validation Acc. ({model_acc*100:.0f}%)')
            ax.fill_between(epochs, mean-std, mean+std, alpha=0.3)
            #ax.scatter(epochs, accuracy, label='val. accuracy (%)', marker=marker, alpha=0.7)
    else:
        if 'training_loss' in plot:
            ax.scatter(epochs, df['train_loss'], label='Train Loss', marker=marker, alpha=alpha)
        if 'validation_loss' in plot:
            validation_min = df['validation_loss'][df['validation_loss'].idxmin]
            ax.scatter(epochs, df['validation_loss'], label=f'Validation Loss ({validation_min:.3f})', marker=marker, alpha=alpha)
        if 'accuracy' in plot:
            model_acc = accuracy[df['validation_loss'].idxmin]
            ax.scatter(epochs, accuracy, label=f"Validation Acc. ({model_acc*100:.0f}%)", marker=marker, alpha=0.7)
        
    ax.set_title(f"{config['model']['base']} w. {config['data']['dataset']}: {config['training']['optimizer']} @{config.getfloat('training', 'learning_rate')*lr_modifier}")
    if ylim:
        ax.set_ylim(ylim)
    ax.legend()
    fig.show()