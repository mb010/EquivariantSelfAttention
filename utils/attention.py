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
def attentions_func(
    batch_of_images,
    net,
    mean=True,
    device=torch.device('cpu'),
    layer_name_base='compatibility_score',
    layer_no=2):
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
            imap, fmap = feature_extractor.forward(images[iteration:iteration+1].float().to(device))

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

# ==========================================================
# Attention Epoch Plot
def AttentionImagesByEpoch(
    sources,
    folder_name,
    net,
    epoch=2000,
    device=torch.device('cpu'),
    layer_name_base='compatibility_score',
    layer_no=2,
    mean=True):
    """
    Args:
        sources: list of Images with type==torch.tensor, of dimension (-1,1,150,150)
        folder_name: directory of saved .pt parameters to load into our network.
    dependancies:
        attentions_func()
        HookedBasedFeatureExtraction() (from within attention_func)
    out:
        attention_maps_temp: list of arrays of all attention maps according to the epoch they were generated.
        epoch_updates: list of epoch numbers for the attention map generations.
    """
    assert device in [torch.device('cpu'), torch.device('cuda')], f"Device needs to be in: [torch.device('cpu'), torch.device('cuda')]"
    assert os.path.exists(folder_name), f"Folder input {folder_name} is not a valid folder path."
    
    attention_maps = []
    original_attention_maps = []
    epoch_updates = []
    
    # Load in models in improving order based on the folder name
    for epoch_temp in range(epoch):
        PATH = f'{folder_name}/{epoch_temp}.pt'
        if os.path.exists(PATH):
            net.load_state_dict(torch.load(PATH,map_location=torch.device(device)))
            net.eval()
            # Generate attention maps with attentions_func and save appropriately.
            attentions , original_attentions = attentions_func(
                np.asarray(sources),
                net,
                mean=mean,
                device=device,
                layer_name_base=layer_name_base,
                layer_no=layer_no)
            
            for i in range(attentions.shape[0]):
                # Averaged attention maps of the images selected in the cell above.
                attention_maps.append(attentions[i])
                # Averaged but unsampled attention maps.
                original_attention_maps.append(original_attentions[i]) 
                #List of when the validation loss / attention maps were updated.
                epoch_updates.append(epoch_temp)

    return attention_maps, original_attention_maps, epoch_updates


# ==========================================================
# Plot for attention by epoch (calls AttentionImagesByEpoch)
def attention_epoch_plot(
    net,
    folder_name,
    source_images,
    logged=False,
    width=5,
    device=torch.device('cpu'),
    layer_name_base='attention',
    layer_no=2,
    cmap_name='magma',
    figsize=(100,100)):
    """
    Function for plotting clean grid of attention maps as they 
    develop throughout the learning stages.
    Args:
        The attention map data,
        original images of sources
        number of unique sources,
        if you want your image logged,
        number of output attentions desired (sampled evenly accross available space)
        epoch labels of when the images were extracted
    Out:
        plt of images concatenated in correct fashion
    """
    
    # cmap_name and RGB potential
    if cmap_name=='RGB':
        mean_ = False
        cmap_name='magma'
    
    
    # Generate attention maps for each available Epoch
    attention_maps_temp, og_attention_maps, epoch_labels = AttentionImagesByEpoch(
        source_images,
        folder_name,
        net,
        epoch=2000,
        device=device,
        layer_name_base=layer_name_base,
        layer_no=layer_no,
        mean=mean_
    )
    
    # Extract terms to be used in plotting
    sample_number = source_images.shape[0]
    no_saved_attentions_epochs = np.asarray(attention_maps_temp).shape[0]//sample_number
    attentions = np.asarray(attention_maps_temp)
    imgs=[]
    labels=[]
    width_array = range(no_saved_attentions_epochs)

    if width <= no_saved_attentions_epochs:
        width_array = np.linspace(0, no_saved_attentions_epochs-1, num=width, dtype=np.int32)
    else:
        width = no_saved_attentions_epochs

    # Prepare the selection of images in the correct order as to be plotted reasonably (and prepare epoch labels)
    for j in range(sample_number):
        if logged:
            imgs.append(np.exp(source_images[j].squeeze()))
        else:
            imgs.append(source_images[j].squeeze())
        for i in width_array:
            #print(sample_number,i,j)
            imgs.append(attention_maps_temp[sample_number*i+j])
            try:
                labels[width-1]
            except:
                labels.append(epoch_labels[sample_number*i])

    # Define the plot of the grid of images
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(sample_number, width+1),
                     axes_pad=0.02, # pad between axes in inch.
                     )
    for idx, (ax, im) in enumerate(zip(grid, imgs)):
        # Transpose for RGB image
        if im.shape[0]==3:
            im = im.transpose(1,2,0)
        # Plot image
        if logged:
            ax.imshow(np.log(im), cmap=cmap_name)
        else:
            ax.imshow(im, cmap=cmap_name)
        # Plot contour if image is source image
        if idx%(width+1)==0:
            ax.contour(im, 1, cmap='cool', alpha=0.5)
        ax.axis('off')
    print(f'Source images followed by their respective averaged attention maps at epochs:\n{labels}')
    plt.show()