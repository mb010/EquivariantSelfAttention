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

from sklearn.metrics import classification_report, roc_curve, auc

# Figures to save:
mp4_plot           = True
distribution_plots = True
training_plot      = True
individual_plot    = True

# Parse config
args        = utils.parse_args()
config_name = args['config']
config      = ConfigParser.ConfigParser(allow_no_value=True)
config.read(f"configs/{config_name}")

# Set seeds for reproduceability
torch.manual_seed(42)
np.random.seed(42)
# Get correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data sets to iterate over
data_configs = [
    "e2attentionmirabest.cfg", # Mirabest Dataset - MBFR
    "e2attentionmingo.cfg" # Mingo Dataset - MLFR
]

# Evaluation augmentations to iterate over
augmentations = [
    "rotation and flipping",
    "random rotation",
    "restricted random rotation"
]

FIG_PATH = config['output']['directory'] +'/'+ config['data']['augment'] +'/'
csv_path = config['output']['directory'] +'/'+ config['data']['augment'] +'/'+ config['output']['training_evaluation']
df = pd.read_csv(csv_path)
best = df.iloc[list(df['validation_update'])].iloc[-1]
# Is this my primary load call? Why am I not using utils.load_model:
#model = utils.utils.load_model(config, load_model='best', device=device, path_supliment=path_supliment)

# Extract models kernel size
if config.has_option('model', 'kernel_size'):
    kernel_size = config.getint('model', 'kernel_size')
elif "LeNet" in config['model']['base']:
    kernel_size = 5
else:
    kernel_size = 3

net = locals()[config['model']['base']](**config['model']).to(device)

################################################################################
### Pixel Attention Distribution ###
# Find or remake code for distrubtion of pixel values.

################################################################################
### MP4 Figure ###
# Make 2d figure for MP4 equivalent explanation for a paper.
# (Compare with Anna's papers & FAIR ViT papers?)


################################################################################
### Individual Source Maps ###
# Select interesting / exemplative sources (in test set) for each data set
# Plot throughout training - Probably encompases the most information.
# Plot individually
if individual_plot:
    interesting_sources_dict = {
        'mirabest': [3,32,30],
        'mingo': [8,16,30],
        'mirabest_labels': [0,1,1],
        'mingo_labels': [0,1,0]
    }
    for cfg in data_configs:
        for data_name in ['mirabest', 'mingo']:
            if data_name in cfg:
                sources_of_interest = interesting_sources_dict[data_name]
        test_data = utils.data.load(config, train=False, augmentation='None', data_loader=False)    # Prepare sources and labels
        for i in sources_of_interest:
            amaps, amap_originals = utils.attention.attentions_func(
                    rotated_input,
                    model,
                    device=device,
                    layer_no=3,
                    layer_name_base='attention',
                    mean=mean
            )
            fig, ax = plt.subplots(1, 2, figsize=(8,8), fontsize=16)
            ax[0].imshow(test_data[i])
            ax[0].set_xticks([])
            ax[0].set_yticks([])

            test_data[i]

################################################################################
### MP4 ###
# Select at most 2 sources for each data set to set an example
# https://github.com/QUVA-Lab/e2cnn/blob/master/visualizations/animation.py
if mp4_plot:
    pass    import numpy as np
    from e2cnn.nn import *
    from e2cnn.group import *
    from e2cnn.gspaces import *

    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    import matplotlib.animation as manimation

    from skimage.transform import resize
    import scipy.ndimage

    import torch
    from typing import Union

    plt.rcParams['image.cmap'] = 'magma'
    plt.rcParams['axes.titlepad'] = 30


    # the irrep of frequency 1 of SO(2) produces the usual 2x2 rotation matrices
    rot_matrix = SO2(1).irrep(1)

    def draw_scalar_field(axs, scalarfield, r: int, contour):
        r'''
        Draw a scalar field
        '''

        D = 3

        m, M = scalarfield.min(), scalarfield.max()

        R = scalarfield.shape[0]
        angle = r * 2 * np.pi / R
        scalarfield = scalarfield[r, ...].squeeze()

        axs[0].clear()
        axs[0].imshow(domask(scalarfield.repeat(D, axis=0).repeat(D, axis=1), margin=8, fmt="image"), clim=(m, M))
        axs[0].contour(contour, 0, levels=[0.], cmap='cool')
        axs[0].set_title("Avg. Attention Map", fontdict={'fontsize': 30})

        stable_view = scipy.ndimage.rotate(scalarfield, -angle * 180.0 / np.pi, (-2, -1), reshape=False, order=2)

        axs[1].clear()
        axs[1].imshow(domask(stable_view.repeat(D, axis=0).repeat(D, axis=1), margin=8, fmt="image"), clim=(m, M))
        axs[1].contour(contour, 0, levels=[0.], cmap='cool')
        axs[1].set_title("Stabilized View", fontdict={'fontsize': 30})
        print(f"contour shape: {contour.shape}")
        print(f"image2 shape: {domask(scalarfield.repeat(D, axis=0).repeat(D, axis=1), margin=8, fmt='image').shape}")
        print(f"image2 shape: {domask(stable_view.repeat(D, axis=0).repeat(D, axis=1), margin=8, fmt='image').shape}")

    def build_mask(s: int, margin: float = 2., dtype=torch.float32):
        mask = torch.zeros(1, 1, s, s, dtype=dtype)
        c = (s - 1) / 2
        t = (c - margin / 100. * c) ** 2
        sig = 2.
        for x in range(s):
            for y in range(s):
                r = (x - c) ** 2 + (y - c) ** 2
                if r > t:
                    mask[..., x, y] = np.exp((t - r) / sig ** 2)
                else:
                    mask[..., x, y] = 1.
        return mask


    def domask(x: Union[np.ndarray, torch.Tensor], margin=2, fmt="torch"):
        if fmt == "image":
            s = x.shape[0]
            mask = build_mask(s, margin)
            mask = mask.permute(0, 2, 3, 1).squeeze()
        else:
            s = x.shape[2]
            mask = build_mask(s, margin)

        if isinstance(x, np.ndarray):
            mask = mask.numpy()

        # use an inverse mask to create a white background (value = 1) instead of a black background (value = 0)
        return mask * x + 1. - mask


    def animate(model: EquivariantModule,
                image: Union[str, np.ndarray],
                outfile: str,
                drawer: callable,
                R: int = 72,
                S: int = 150,
                duration: float = 10.,
                figsize=(21, 10),
                RGB: bool=False,
                ):
        r'''

        Build a video animation

        Args:
            model: the equivariant model
            image: the input image
            outfile: name of the output file
            drawer: method which plots the output field. use one of the methods ``draw_scalar_field``, ``draw_vector_field`` or ``draw_mixed_field``
            R: number of rotations of the input to render, i.e. number of frames in the video
            S: size the input image is downsampled to before being fed in the model
            duration: duration (in seconds) of the video
            figsize: shape of the video (see matplotlib.pyplot.figure())
            RGB: Whether or not to output attention maps as RGB images
        '''

        fig, axs = plt.subplots(1, 3, figsize=figsize)
        fig.set_tight_layout(True)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        model.eval()

        if isinstance(image, str):
            image = mpimg.imread(image).transpose((2, 0, 1))
        image = image[np.newaxis, :, :, :]
        _, C, w, h = image.shape

        # resize the image to have a squared shape
        # the image is initially upsampled to (up to) 4 times the specified size S
        # rotations are performed at this higher resolution and are later downsampled to size S
        # this helps reducing the interpolation artifacts for rotations which are not multiple of pi/2
        T = max(4 * S + 1, 513)
        image = resize(image, (1, C, T, T), anti_aliasing=True)
        C = np.min([C, 3])

        original_inputs = []

        for r in range(R):
            # Rotate the image
            # N.B.: this only works for trivial (i.e. scalar) input fields like RGB images.
            # In case vector fields are used in input, one should also rotate the channels using the group representation
            # of the corresponding FieldType
            rot_input = scipy.ndimage.rotate(image, r * 360.0 / R, (-2, -1), reshape=False, order=2)

            # discard non-RGB channels
            rot_input = rot_input[:, :C, ...]

            original_inputs.append(rot_input)

        original_inputs = np.concatenate(original_inputs, axis=0)

        # mask the input images to remove the pixels which would be moved outside the grid by a rotation
        original_inputs *= build_mask(T, margin=5.2).numpy()

        # downsample the images
        inputs = resize(original_inputs, (original_inputs.shape[0], C, S, S), anti_aliasing=True)

        rotated_input = torch.tensor(inputs, dtype=torch.float32)
        rotated_input *= build_mask(S, margin=5.2)### ????

        # normalize the colors of the images before feeding them into the model
        rotated_input -= rotated_input[0, ...].view(C, -1).mean(dim=1).view(1, C, 1, 1)
        rotated_input /= rotated_input[0, ...].view(C, -1).std(dim=1).view(1, C, 1, 1)

        del inputs

        rotated_input = rotated_input.to(device)
        # wrap the tensor in a GeometricTensor
        #rotated_input = GeometricTensor(rotated_input, model.in_type)


        # pass the images through the model to compute the output field
        with torch.no_grad():

            # In training mode, the batch normalization layers normalize the features with the batch statistics
            # This sometimes produces nicer output fields
            # model.train()
            mean = not RGB
            #output = model(rotated_input)
            amaps, amap_originals = utils.attention.attentions_func(
                    rotated_input,
                    model,
                    device=device,
                    layer_no=3,
                    layer_name_base='attention',
                    mean=mean
            )

        # mask the inputs with a white background for visualization purpose
        original_inputs = domask(original_inputs, margin=5)
        rotated_input = domask(rotated_input.cpu().numpy(), margin=2)
        #amaps = amaps.transpose(0, 2, 3, 1)

        title_suppliment = "Avg. " if not RGB else ""

        # save each rotated image and its corresponding output in a different frame of the video
        for r in range(R):
            n = str(r)
            while len(n)<3:
                n = '0'+n

            # render the input image
            source = rotated_input[r, ...].transpose(1, 2, 0).squeeze()

            axs[0].clear()
            axs[0].imshow(source)
            axs[0].contour(source, 0, levels=[0.], cmap='cool')
            axs[0].set_title("Input", fontdict={'fontsize': 30})

            # render the output and the stabilized output
            #drawer(axs[1:], amaps, r, contour=source)

            m, M = amaps.min(), amaps.max()
            if RGB:
                amap = domask(amaps[r, ...].squeeze(), margin=2)
                amap = amap.squeeze().transpose(1, 2, 0)
            else:
                # If you want non masked, comment out the below
                amap = domask(amaps[r, ...].squeeze(), margin=2, fmt="image")

            axs[1].clear()
            axs[1].imshow(amap, clim=(m, M))
            #axs[1].contour(source, 0, levels=[0.], cmap='cool')
            axs[1].set_title(f"{title_suppliment}Attention Map", fontdict={'fontsize': 30})

            angle = r * 2 * np.pi/R
            stable_view = scipy.ndimage.rotate(amaps[r, ...].squeeze(), -angle * 180.0 / np.pi, (-2, -1), reshape=False, order=2)
            if RGB:
                stable_view = domask(stable_view, margin=2)
                stable_view = stable_view.squeeze().transpose(1, 2, 0)
            else:
                # If you want non masked, comment out the below
                stable_view = domask(stable_view, margin=2, fmt="image")

            axs[2].clear()
            axs[2].imshow(stable_view, clim=(m, M))
            axs[2].contour(rotated_input[0, ...].transpose(1, 2, 0).squeeze(), 0, levels=[0.], cmap='cool')
            axs[2].set_title("Stabilized View", fontdict={'fontsize': 30})

            for ax in axs:
                ax.axis('off')

            fig.set_tight_layout(True)
            plt.savefig(outfile.replace("*", n))
    output = "scalar"

    # Load in my equivariant model
    cfg = "5kernel_e2attentionmingo-RandAug.cfg"
    cfg = "C8_attention_mirabest.cfg"
    cfg = "configs/"+cfg
    config = ConfigParser.ConfigParser(allow_no_value=True)
    config.read(cfg)
    path_supliment = config['data']['augment']
    if path_supliment in ['True', 'False']:
        path_supliment=''
    else:
        path_supliment+='/'
    model = utils.utils.load_model(config, load_model='best', device=device, path_supliment=path_supliment)

    # Load in test data / image
    interesting_sources = [ # Must be sorted by total number for correct labelling [batch no, source no within given batch]
        [0,12],
        [0,13],
        [1,0],
        [1,5],
        [1,13],
        [1,14],
        [5,10]
    ]

    interesting_sources = [
        [1,0]
    ]

    interesting_sources = [
        [0,0],
        [0,3]
    ]

    imgs = []
    labels = []
    test_data_loader = utils.data.load(config, train=False, augmentation='None', data_loader=True)    # Prepare sources and labels
    for idx, (test_data, l) in enumerate(test_data_loader):
        for idy, img in enumerate(test_data):
            if [idx,idy] in interesting_sources:
                imgs.append(img)
                labels.append(l[idy])

    frames = 24*6
    gif_duration_s = 6
    duration = gif_duration_s/frames*1000 # duration of single image within gif (ms)

    completed = []
    for idx, image in enumerate(imgs):
        for color in ['']:#,'RGB']:
            source_no = 16*interesting_sources[idx][0] + interesting_sources[idx][1]
            RGB_ = True if color=='RGB' else False
            if source_no in completed:
                pass
            else:
                print(f"starting for source {source_no} {color}")
                out_path = f"/raid/scratch/mbowles/EquivariantSelfAttention/figures/animation_frames/{color}amap_e2MiraBest{source_no}_FR{labels[idx]+1}_*.png"
                # build the animation
                animate(model, image, out_path, draw_scalar_field, R=frames, S=150, RGB=RGB_)

                # produce final video using ffmpeg command:
                #ffmpeg -framerate 24 -i animation_frames/amap_MingoLotSS16_FR2_%03d.png -pix_fmt yuv420p FILE_NAME.mp4
                #fp_out = f"/raid/scratch/mbowles/EquivariantSelfAttention/figures/{color}amap_MingoLotSS{source_no}_FR{labels[idx]+1}.gif"
                # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
                #img, *imgs = [Image.open(f) for f in sorted(glob.glob(out_path))]
                #img.save(fp=fp_out, format='GIF', append_images=imgs,
                #         save_all=True, duration=duration, loop=0)


################################################################################
### Various Distribution Plots ###
# Decide on exact format for all calls. Maybe cmap='seismic' (harder split)
# Add save path parameter to call
# Add plot_confusion_matrix to call (distributions)
if distribution_plots:
    repititions = 72

    # Load in model
    path_supliment = config['data']['augment']
    if path_supliment in ['True', 'False']:
        path_supliment=''
    else:
        path_supliment+='/'

    # Load in data
    test_data_loader = utils.data.load(data_config, train=False, augmentation='random rotation', data_loader=True)    # Prepare sources and labels
    for idx, r in tqdm(enumerate(range(repititions))):
        for idy, (test_data, l) in enumerate(test_data_loader):
            # Produce Attention Maps
            amap_, amap_originals_ = utils.attention.attentions_func(
                    test_data,
                    model,
                    device=device,
                    layer_no=3,
                    layer_name_base='attention',
                    mean=False
            )
            model.eval()
            raw_predictions_ = model(test_data.to(device))

            # Save data for analysis
            if idx+idy==0:
                sources = test_data.numpy()
                labels = np.asarray(l)
                raw_predictions = raw_predictions_.cpu().detach().numpy()
                amap = amap_
                original_0 = np.concatenate(amap_originals_[0::3], axis=2)
                original_1 = np.concatenate(amap_originals_[1::3], axis=2)
                original_2 = np.concatenate(amap_originals_[2::3], axis=2)
            else:
                sources = np.append(sources, test_data.numpy(), axis=0)
                labels = np.append(labels, np.asarray(l), axis=0)
                raw_predictions = np.append(raw_predictions, raw_predictions_.cpu().detach().numpy(), axis=0)
                amap = np.append(amap, amap_, axis=0)
                original_0 = np.append(original_0, np.concatenate(amap_originals_[0::3], axis=2), axis=2)
                original_1 = np.append(original_1, np.concatenate(amap_originals_[1::3], axis=2), axis=2)
                original_2 = np.append(original_2, np.concatenate(amap_originals_[2::3], axis=2), axis=2)


    predictions = raw_predictions.argmax(axis=1)

    print(raw_predictions.shape)
    print(labels.shape)
    print(sources.shape)
    print(amap.shape)
    print(predictions.shape)


    # Attention Map Differences
    plot_3(
        fri_amap_sum.mean(0)*mask, frii_amap_sum.mean(0)*mask, fr_amap_diff.mean(0)*mask,
        cmaps=['coolwarm'], titles=["Attention FRI Mean", "Attention FRII Mean", "Attention Difference"],
        cbars_bool=[True], figsize=(16,9),
        vmin=['adaptive', 'adaptive', "adaptive"],
        vmax=['adaptive', 'adaptive', "adaptive"],
        contour=fr_source_diff.squeeze()*mask
    )
    # Attention Map Differences by Gate
    plot_3(
        fri_amap_sum[0]*mask,
        frii_amap_sum[0]*mask,
        fr_amap_diff[0]*mask,
        cmaps=['coolwarm'],
        titles=["Attention FRI[0] Mean", "Attention FRII[0] Mean", "Attention Difference[0]"],
        cbars_bool=[True], figsize=(16,9),
        vmin=['adaptive', 'adaptive', "adaptive"],
        vmax=['adaptive', 'adaptive', "adaptive"],
        contour=fr_source_diff.squeeze()*mask
    )
    plot_3(
        fri_amap_sum[1]*mask,
        frii_amap_sum[1]*mask,
        fr_amap_diff[1]*mask,
        cmaps=['coolwarm'],
        titles=["Attention FRI[1] Mean", "Attention FRII[1] Mean", "Attention Difference[1]"],
        cbars_bool=[True], figsize=(16,9),
        vmin=['adaptive', 'adaptive', "adaptive"],
        vmax=['adaptive', 'adaptive', "adaptive"],
        contour=fr_source_diff.squeeze()*mask
    )
    plot_3(
        fri_amap_sum[2]*mask,
        frii_amap_sum[2]*mask,
        fr_amap_diff[2]*mask,
        cmaps=['coolwarm'],
        titles=["Attention FRI[2] Mean", "Attention FRII[2] Mean", "Attention Difference[2]"],
        cbars_bool=[True],
        figsize=(16,9),
        vmin=['adaptive', 'adaptive', "adaptive"],
        vmax=['adaptive', 'adaptive', "adaptive"],
        contour=fr_source_diff.squeeze()*mask
    )

################################################################################
### Training Plot ###
# Change decimal places in title value
# Add save path parameter to call
if training_plot:
    ylim = [0,1]
    ylim = None
    window_size=10

    for config_name_ in config_names:
        config_name = "configs/"+config_name_

        config = ConfigParser.ConfigParser(allow_no_value=True)
        config.read(config_name)

        path_supliment = config['data']['augment']
        if path_supliment in ['True', 'False']:
            path_supliment=''
        else:
            path_supliment += '/'

        utils.evaluation.training_plot(
            config,
            ylim=ylim,
            plot=['training_loss', 'validation_loss', 'accuracy'],
            lr_modifier=1,
            path_supliment=path_supliment,
            mean=True,
            window_size=window_size
        )

### evaluation ###
"""for d_cfg in data_configs:
    for augmentation in augmentations:
        path_supliment = config['data']['augment']+'/'
        model = utils.utils.load_model(config, load_model='best', device=device, path_supliment=path_supliment)
        data_config.read('configs/'+d_cfg)
        print(f"Evaluating {cfg}: {config['output']['directory']}/{config['data']['augment']}\t{data_config['data']['dataset']}\t{augmentation}")
        data  = utils.data.load(
            data_config,
            train=False,
            augmentation=augmentation,
            data_loader=True
        )

        y_pred, y_labels = utils.evaluation.predict(
            model,
            data,
            augmentation_loops=100,
            raw_predictions=True,
            device=device,
            verbose=True
        )

        utils.evaluation.save_evaluation(
            y_pred,
            y_labels,
            model_name=config['model']['base'],
            kernel_size=kernel_size,
            train_data=config['data']['dataset'],
            train_augmentation=config['data']['augment'],
            test_data=data_config['data']['dataset'],
            test_augmentation=augmentation,
            epoch=int(best.name),
            best=True,
            raw=False,
            PATH='/share/nas/mbowles/EquivariantSelfAttention/' + config['output']['directory'] +'/'+ config['data']['augment']
        )
"""