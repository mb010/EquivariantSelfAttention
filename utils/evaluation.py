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
from tqdm import tqdm

# Import various network architectures
from networks import AGRadGalNet, DNSteerableLeNet, DNSteerableAGRadGalNet #e2cnn module only works in python3.7+
# Import various data classes
from datasets import FRDEEPF
from datasets import MiraBest_full, MBFR, MBFRConfident, MBFRUncertain, MBHybrid
from datasets import MingoLoTSS, MLFR, MLFRTest
from torchvision.datasets import MNIST

# sklearn
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Set seeds for reproduceability
torch.manual_seed(42)
np.random.seed(42)

# ==========================================================
def predict(model,
            data,
            augmentation_loops=1,
            raw_predictions=True,
            device = torch.device('cpu'),
            verbose=False
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
    f = tqdm if verbose else lambda x:x
    model = model.to(device)
    for loop in f(range(augmentation_loops)):
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

# ==========================================================
def save_evaluation(y_pred,
                    y_labels,
                    model_name='',
                    kernel_size=3,
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
    # Use n_classes to determine how to deal with predictions
    n_classes = 2 if test_data!='MNIST' else 10
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

        if n_classes == 2:
            fpr, tpr, thresholds = roc_curve(y_labels.flatten(), pred[:,1].numpy())
            AUC = auc(fpr, tpr)
        else:
            # Compute ROC curve and ROC area for each class
            fpr = {}
            tpr = {}
            thresholds = {}
            roc_auc = {}
            # This would be useful for plotting
            # Surpassing for quick look data using micro-averaging
            #for i in range(n_classes):
            #    tmp_labels = np.where(y_labels.flatten()==i, 1, 0)
            #    fpr[i], tpr[i], thresholds[i] = roc_curve(tmp_labels, pred[:, i])
            #    roc_auc[i] = auc(fpr[i], tpr[i])
            fpr["micro"], tpr["micro"], thresholds["micro"] = roc_curve(y_lables.ravel(), y_score.ravel())
            AUC = auc(fpr["micro"], tpr["micro"])

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
            'kernel_size': [kernel_size],
            'train_data': [train_data],
            'train_augmentation': [train_augmentation],
            'test_data': [test_data],
            'test_augmentation': [test_augmentation],
            'epoch': [epoch],
            'best': [best],
            'auc': [AUC]
        }
        for i in range(n_classes):
            evaluation[f"{i} precision"] = [report[str(i)]['precision']]
            evaluation[f"{i} recall"]    = [report[str(i)]['recall']]
            evaluation[f"{i} f1-score"]  = [report[str(i)]['f1-score']]
            evaluation[f"{i} support"]   = [report[str(i)]['support']]

        df = pd.DataFrame.from_dict(evaluation)
        if os.path.isfile(PATH+'full_evaluations.csv'):
            df.to_csv(PATH+"full_evaluations.csv", mode='a', index=False, header=False)
        else:
            df.to_csv(PATH+"full_evaluations.csv", mode='w', index=False, header=True)

# ==========================================================
def training_plot(
    config,
    plot=['validation_loss', 'training_loss', 'accuracy'],
    path_supliment='',
    marker='o',
    alpha=0.7,
    lr_modifier=1,
    fontsize=20,
    ylim=None,
    mean=False,
    window_size=1,
    save=False):

    csv_path = config['output']['directory'] +'/'+ path_supliment + config['output']['training_evaluation']
    df = pd.read_csv(csv_path)
    epochs = np.asarray(range(len(df)))
    accuracy = df.validation_accuracy#(df.TP + df.TN)/(df.TP+df.TN + df.FP+df.FN)

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

    ax.set_title(f"{config['model']['base']} w. {config['data']['dataset']}: {config['training']['optimizer']} @{config.getfloat('training', 'learning_rate')*lr_modifier:.1E}")
    if ylim:
        ax.set_ylim(ylim)
    ax.legend()

    if save:
        plt.savefig(save, transparent=True)
    else:
        fig.show()

# ==========================================================
# ROC Curve
def plot_roc_curve(fpr, tpr, title=None, save=False):
    fig, ax = plt.subplots(figsize=(8,8))
    if type(fpr) != dict:
        AUC = auc(fpr,tpr)
        ax.plot(fpr, tpr, linewidth='2', label=f"ROC Curve with AUC={AUC:.3f}")
    else:
        AUC = {}
        for key in fpr.keys():
            AUC[key] = auc(fpr[key],tpr[key])
            ax.plot(fpr[key], tpr[key], linewidth='2', label=f"{key}: AUC={AUC[key]:.3f}")

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    if title==None:
        pass
    else:
        ax.set_title(title)
    ax.grid(True, 'large')
    ax.set_xlim(None,1)
    ax.set_ylim(0,None)
    if not save:
        fig.show()
    else:
        fig.savefig(save)

# ==========================================================
# Binary Confusion Matrix
def plot_conf_mat(conf_matrix, normalised=True, n_classes=2, format_input=None, title='Confusion Matrix', publication=False, save=''):
    # Following along the lines of (from the github on 29.04.2020)
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    plt.rcParams.update({'font.size': 14})

    classes = ['FRI','FRII']
    xticks_rotation='horizontal'
    matrix = conf_matrix.copy() #Otherwise can change matrix inplace, which is undesirable for potential further processing.
    temp = np.asarray(matrix)
    values_format = '.4g'
    if normalised==True:
        values_format = '.1%'
        for i in range(matrix.shape[0]):
            matrix = matrix.astype('float64')
            if publication:
                matrix[i] = matrix[i]/matrix[i].sum()
            else:
                matrix[i] = matrix[i]/matrix[i].sum()

    if type(format_input) == str:
        values_format = format_input

    # Initialise figure
    fig, ax = plt.subplots(figsize=(8,8))
    img = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(img,ax=ax)
    cmap_min, cmap_max = img.cmap(0), img.cmap(256)

    # print text with appropriate color depending on background
    text = np.empty_like(matrix, dtype=object)
    thresh = (matrix.max() + matrix.min()) / 2.0
    for i, j in product(range(n_classes), range(n_classes)):
        color = cmap_max if matrix[i, j] < thresh else cmap_min
        text[i, j] = ax.text(j, i, format(matrix[i, j], values_format),
                             ha="center", va="center",
                             color=color)
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=classes,
           yticklabels=classes,
           ylabel="True label",
           xlabel="Predicted label")
    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)
    plt.title(title)
    plt.show()

    if save!='':
        plt.savefig(save)
