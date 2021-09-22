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
from tqdm.notebook import trange, tqdm
from scipy.special import logsumexp

import utils
import pickle
# Ipmport various network architectures
from networks import AGRadGalNet, DNSteerableLeNet, DNSteerableAGRadGalNet #e2cnn module only works in python3.7+
# Import various data classes
from datasets import FRDEEPF
from datasets import MiraBest_full, MBFRConfident, MBFRUncertain, MBHybrid
from datasets import MingoLoTSS, MLFR, MLFRTest
from tqdm import tqdm

#Ensure the GPU is being used for the calculation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




#Copies over the weights of the network
def WeightTransfer(referenceModel, newModel):
    newModel.parameters = referenceModel.parameters
    #Clear the model were copying weights from to clear up GPU memory.
    del(referenceModel)
    #Set weights of the newModel to be untrainable apart from last fully connected layer
    for name, param in newModel.named_parameters():
        param.requires_grad=False
    newModel.last_weights().weight.requires_grad=True
    return;
    
# Make the math imports to calculate metrics related to the Fisher Information, such as the Jacobian/Hessian
def jacobian(f, w, create_graph=False):   

    """
    function to find the jacobian of f with respect to x parameters of model

    output has shape (len(f), len(x))
    """

    jac = []
    output = f
    #f = torch.log(f)
    grad_f = torch.zeros_like(f)
    
    for i in range(len(f)):                                                                      
        grad_f[i] = 1.
        grad_f_x = torch.autograd.grad(f, w, grad_f, retain_graph=True, create_graph=create_graph, allow_unused=True)
        J = torch.cat(grad_f_x).view(-1)
        jac.append(J * torch.sqrt(output[i]))
        grad_f[i] = 0.
                                                  
    return torch.stack(jac).reshape(f.shape + J.shape)
#---MAIN FISHER INITALISATION---#
'''
net --> PyTorch Compatiable Model
train_loader --> PyTorch/SKLearn Compatiable DataLoader Function
n_iterations --> Number of realisations of the fisher matrix
'''
def CalcFIM(net, train_loader, n_iterations, outputsize, workingdir, approximation=None):
    #First we need to obtain the number of weights in the last fully connected layer and freeze the weights in every other layer
    Rank = []
    FR = []
    print(f"Calculating {n_iterations} of the Fisher...")
    Number_of_FisherIts = n_iterations;
    n_weight = outputsize * len(net.last_weights().weight[0])
    
    realisations_torch = torch.zeros((Number_of_FisherIts,n_weight,n_weight)) #All fishers
    for i in tqdm(range(Number_of_FisherIts)):
        torch.nn.init.uniform_(net.last_weights().weight, -1., 1.)
        #torch.nn.init.normal_(net.classifier.weight, mean=0.0, std=1.0)
        #torch.nn.init.xavier_uniform_(net.classifier.weight, gain=1.0)
        #torch.nn.init.kaiming_normal(net.classifier.weight)
        #torch.nn.init.orthogonal_(net.classifier.weight, gain=1.0)
        w = net.last_weights().weight
        flat_w = w.view(-1).cpu()
        fisher = torch.zeros((n_weight,n_weight)).to('cuda')
        for x_n, y_n in train_loader:
            x_n, y_n = x_n.to('cuda'), y_n.to('cuda')
            f_n = net(x_n)
            logit = f_n
            f_n = F.softmax(f_n, dim=1)
            for row, logitrow in zip(f_n, logit):
                if (approximation=="emperical"):
                    pi_n = row
                    diag_pi_n = torch.diag(pi_n.squeeze(0)).to('cuda')
                    pi_n_pi_n_T=torch.outer(pi_n,torch.t(pi_n))
                    J_f = jacobian((torch.squeeze(row,0)),w)
                    J_f_T = J_f.permute(1,0)
                    K2 = diag_pi_n-pi_n_pi_n_T
                    K3 = K2.cuda()
                    fisher += torch.matmul((torch.matmul(J_f_T,K3)),J_f)
                else:
                    J_f = jacobian((torch.squeeze(row,0)),w)
                    grads = J_f
                    #grads = jacob1.detach().numpy()
                    temp_sum = torch.zeros((outputsize, n_weight, n_weight)).to('cuda')
                    for j in range(outputsize):
                        temp_sum[j] += torch.outer(grads[j], torch.t(grads[j]))
                    fisher += torch.sum(temp_sum, axis=0)
        with torch.no_grad():
            try:
                rank = torch.matrix_rank(fisher).item()
                Rank.append(rank)
                realisations_torch[i] = fisher.cpu()
                Fw = np.matmul(fisher.cpu().numpy(),flat_w)
                wFw = np.dot(flat_w,Fw)
                FR.append(wFw)
            except:
                pass;
        if (i%1000)==0:
            print(f"Saving Fisher Realisations to a Pickle File for iteration {i}...")
            pickle.dump(realisations_torch, open(f"{workingdir}/fishers.p", "wb"))
            print(f"Saving FR Norm Realisations to a Pickle File for iteration {i}")
            pickle.dump(FR, open(f"{workingdir}/raonorm.p", "wb"))
    return realisations_torch, Rank, FR;


def Hessian(y, x):
    hessian = Jacobian(Jacobian(y, x, create_graph=False), x)
    return hessian;

#Functions to evaluate the Fisher Metrics, such as Eigenvalue and Rank
def calc_eig(Fishers):
    for fisher in Fishers:
        eigval = torch.eig(fisher, eigenvectors=False,  out=None).eigenvalues[:,0]
    return eigval
    
def normalise(Fishers, n_params):
    fisher_trace = np.trace(np.average(Fishers, axis=0)) 
    finalfisher = np.average(Fishers, axis=0)
    fhat = finalfisher/fisher_trace
    return fhat * n_params

#Functions to calculate the Effective Dimension as computed in https://zenodo.org/record/4732830 by Amira Abbas
def effective_dimension(model, f_hat, num_thetas, n, outputs):
    '''
    model -> pytorch compatible model
    f_hat -> normalised array of fisher matrix
    n -> Number of samples for the computation (len trainloader?) in list
    outputs -> Number of classes
    '''
    effective_dim = []

    for ns in tqdm(n):
        Fhat = f_hat * ns / (2 * np.pi * np.log(ns))
        one_plus_F = np.eye(outputs) + Fhat
        det = np.linalg.slogdet(one_plus_F)[1]  # log det because of overflow
        r = det / 2  # divide by 2 because of sqrt
        effective_dim.append(2 * (logsumexp(r) - np.log(num_thetas)) / np.log(ns / (2 * np.pi * np.log(ns))))

    return effective_dim