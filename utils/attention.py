import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from itertools import product

import torchvision
from torchvision import transforms, datasets
#from FRDEEP import FRDEEPF
#from MiraBest import MiraBest_full
#from models_new import *
from PIL import Image
import PIL
from torchsummary import summary
import torch.nn as nn
import torch
#from models.networks_other import init_weights
#from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, recall_score, f1_score, precision_score, auc#, plot_confusion_matrix
from skimage.transform import resize
from mpl_toolkits.axes_grid1 import ImageGrid

# initially from networks.utils
class HookBasedFeatureExtractor(nn.Module):
    def __init__(self, submodule, layername, upscale=False):
        super(HookBasedFeatureExtractor, self).__init__()

        self.submodule = submodule
        self.submodule.eval()
        self.layername = layername
        self.outputs_size = None
        self.outputs = None
        self.inputs = None
        self.inputs_size = None
        self.upscale = upscale

    def get_input_array(self, m, i, o):
        if isinstance(i, tuple):
            self.inputs = [i[index].data.clone() for index in range(len(i))]
            self.inputs_size = [input.size() for input in self.inputs]
        else:
            self.inputs = i.data.clone()
            self.inputs_size = self.input.size()

    def get_output_array(self, m, i, o):
        if isinstance(o, tuple):
            self.outputs = [o[index].data.clone() for index in range(len(o))]
            self.outputs_size = [output.size() for output in self.outputs]
        else:
            self.outputs = o.data.clone()
            self.outputs_size = self.outputs.size()

    def rescale_output_array(self, newsize):
        us = nn.Upsample(size=newsize[2:], mode='bilinear')
        if isinstance(self.outputs, list):
            for index in range(len(self.outputs)): self.outputs[index] = us(self.outputs[index]).data()
        else:
            self.outputs = us(self.outputs).data()

    def forward(self, x):
        target_layer = self.submodule._modules.get(self.layername)

        # Collect the output tensor
        h_inp = target_layer.register_forward_hook(self.get_input_array)
        h_out = target_layer.register_forward_hook(self.get_output_array)
        self.submodule(x)
        h_inp.remove()
        h_out.remove()

        # Rescale the feature-map if it's required
        if self.upscale: self.rescale_output_array(x.size())

        return self.inputs, self.outputs

# ==========================================================
# Attention Map Call -- From original Bowles 2021 attention paper.
def attentions_func(batch_of_images,
                    net,
                    mean=True,
                    device=torch.device('cpu'),
                    layer_name_base='compatibility_score',
                    layer_no=2
                   ):
    """
    Args:
        batch_of_images: Images with type==torch.tensor, of dimension (-1,1,150,150)
        net: model being loaded in.
    Calls on: HookBasedFeatureExtractor to call out designed attention maps.
    Output: Upscaled Attention Maps, Attention Maps
    """
    assert device in [torch.device('cpu'), torch.device('cuda')], f"Device needs to be in: [torch.device('cpu'), torch.device('cuda')]"
    assert len(batch_of_images.shape)==4, f'Batch input expected to be of dimensions: BxCxWxH (4 dims)'

    if type(batch_of_images) == type(np.array([])):
        batch_of_images = torch.tensor(batch_of_images)

    images = batch_of_images
    AMap_originals = []
    for iteration in range(batch_of_images.shape[0]): # Should be able to do this without iterating through the batches.
        for i in range(layer_no):
            feature_extractor = HookBasedFeatureExtractor(net, f'{layer_name_base}{i+1}', upscale=False)
            imap, fmap = feature_extractor.forward(images[iteration:iteration+1].to(device))

            if not fmap: #Will pass if fmap is none or empty etc.
                continue #(ie. Skips iteration if compatibility_score{i} does not compute with the network.)

            attention = fmap[1].cpu().numpy().squeeze()
            attention = np.expand_dims(resize(attention, (150, 150), mode='constant', preserve_range=True), axis=2)

            if (i == 0):
                attentions_temp = attention
            else:
                attentions_temp = np.append(attentions_temp, attention, axis=2)
            AMap_originals.append(np.expand_dims(fmap[1].cpu().numpy().squeeze(), axis=2))

        if iteration == 0:
            attentions = np.expand_dims(attentions_temp, 3)
        else:
            attentions = np.append(attentions, np.expand_dims(attentions_temp, 3), axis=3)

    # Channel dimension is compatibility_score1 / compatibility_score2 respectively (take mean for total attention):
    attentions_out = np.reshape(attentions.transpose(3, 2, 0, 1),(-1, layer_no, 150,150))
    if mean:
        # Take the mean over all attention maps
        attentions_out = np.mean(attentions_out, axis=1)

    return attentions_out , AMap_originals