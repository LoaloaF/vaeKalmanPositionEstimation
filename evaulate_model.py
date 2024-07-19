import numpy as np
import pandas as pd
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data_utils 
from torchvision import datasets, transforms
# import plotly.express as px
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
# from skimage.transform import resize
# from skimage.io import imread
import scipy.stats as stats
from tqdm.autonotebook import tqdm, trange
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import warnings
# from catalyst.utils import set_global_seed
from copy import deepcopy
from sklearn.preprocessing import QuantileTransformer
# import plotly.express as px
# sns.set_style('darkgrid')
warnings.filterwarnings("ignore")

np.random.seed(42)
torch.manual_seed(42)

from utils import get_device
from utils import get_dataloaders
from utils import log_likelihood

from VAE import VAE
from UnityLocalizationDataset import UnityLocalizationDataset
import matplotlib.colors as mcolors
from scipy.stats import pearsonr

from kalman_filter import create_arena_canvas
from z_prob_estimation import mle, nlog_likelihood

def plot_pos_errors(device, model, test_loader, zfromH=False):
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["green", "red"])

    fig = plt.figure(figsize=(5, 5))
    fig.suptitle(r'Inferring position from latent $\hat{p} = z^T W$')
    plt.vlines([-50,50], -55,55, edgecolor='k', linestyles='--', alpha=.5)
    plt.hlines([-50,50], -55,55, edgecolor='k', linestyles='--', alpha=.5)
    plt.xlim(-55,55)
    plt.ylim(-55,55)
    plt.grid()
    plt.xlabel("X Position")
    plt.ylabel("Y Position")

    all_distances = []
    # for i, (x_batch, y_batch) in enumerate(train_loader):
    for j, (x_batch, y_batch) in enumerate(test_loader):
    # for x_batch, pos_batch in test_loader:
        # x_batch = x_batch.to(device)
        
        y_batch = y_batch.to(device)
        x_batch = x_batch.to(device)
        # y_batch = y_batch.to(device)
        # print(y_batch[:, :2].min(), y_batch[:, :2].max())
                
        model.eval()
        with torch.no_grad():
            if not zfromH:
                z = model.encode(x_batch)
            else:
                z = model.position2latent_mapping(y_batch)
            y_pred = model.latent2position_mapping(z).cpu().numpy()
        
        # plt.scatter(y_batch[:, 0], y_batch[:, 1], c='k')
        # plt.scatter(y_pred[:, 0], y_pred[:, 1], c='g')
        y_batch = y_batch.cpu().numpy()
        
        y_pred *= 55
        y_batch *= 55
        for i in range(len(y_batch)):
            distance = np.sqrt((y_batch[i, 0] - y_pred[i, 0])**2 + (y_batch[i, 1] - y_pred[i, 1])**2)
            all_distances.append(distance)
            
            
            col =  cmap((distance/20).item())
            # Calculate the starting point (x, y)
            x_start = y_batch[i, 0]
            y_start = y_batch[i, 1]
            # Calculate the changes in x and y (dx, dy) to point towards the prediction
            dx = y_pred[i, 0] - y_batch[i, 0]
            dy = y_pred[i, 1] - y_batch[i, 1]
            # Plot the arrow from actual data point to prediction
            plt.arrow(x_start, y_start, dx, dy, color=col, alpha=0.3, length_includes_head=True, head_width=1, head_length=1)

        
        if j >20:
            break
    plt.savefig("./figs/euclid_distances_arena.svg")
    plt.show()
    
    print(np.mean(all_distances)/20)
    
    plt.figure(figsize=(5,5))
    ax = fig.gca()
    ax.set_facecolor("#eaeaf2ff")
    plt.hist(all_distances, bins=30)
    plt.title("Position estimation distribution")
    plt.xlabel(r"Eucledian distance: $\|\mathbf{p} - \mathbf{\hat{p}}\|$")
    plt.savefig("./figs/euclid_dist_distr_Hp.svg")
    plt.show()
        
def plot_latent_dims(device, model, train_load, test_loader, zfromH=False):
    # plt.figure(figsize=(9, 9))
    fig, axes = plt.subplots(nrows=4, ncols=6, sharex=True, sharey=True, figsize=(8,6))
    fig.suptitle(r"Embedding $m$ distribution")
    axes = axes.flatten()
    
    for ax in axes:
        # ax.axis('off')
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.tick_params(bottom=True, labelbottom=True, left=False, labelleft=False)
    
    all_embeddings = []
    # print(len(test_loader))
    # for i, (x_batch, pos_batch) in enumerate(train_loader):
    for i, (x_batch, pos_batch) in enumerate(test_loader):
    # for x_batch, pos_batch in test_loader:
        x_batch = x_batch.to(device)
        pos_batch = pos_batch.to(device)
        
        model.eval()
        with torch.no_grad():
            if not zfromH:
                z = model.encode(x_batch)
            else:
                z = model.position2latent_mapping(pos_batch)
            all_embeddings.append(z.cpu().numpy())


        if i >10:
            break
    
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(all_embeddings.shape)
    
    for i in range(24):
        axes[i].hist(all_embeddings[:,i], bins=10, edgecolor='white', linewidth=.2)
        
        # use latex to print z_{i}
        axes[i].set_title(rf'$m_{{{i}}}$')
    # plt.savefig("./figs/z_distribution.svg")
    plt.show()
        
def plot_reconstruction_error(device, model, train_loader, test_loader, size=5000):
    # clear_output(wait=True)
    fig, ax = create_arena_canvas()
    # plt.figure(figsize=(7, 5))
    plt.title(r"Image reconstruction error: $\|\mathbf{I} - \mathbf{\hat{I}}\|$")
    
    all_errors = []
    for i, (x_batch, pos_batch) in enumerate(test_loader):
        x_batch = x_batch.to(device)
        # pos_batch = pos_batch.to(device)
        
        model.eval()
        with torch.no_grad():
            mu, logsigma, reconstruction = model(x_batch)
            error_batch = log_likelihood(x_batch, reconstruction, collapse_batch_dim=False).cpu()
        
        all_errors.extend(list(error_batch))
        scatter = plt.scatter(pos_batch[:, 0]*55, pos_batch[:, 1]*55, c=error_batch, s=20,
                              cmap="viridis", vmin=550, vmax=750, alpha=.7, edgecolors='none')

        if i >20:
            break
    plt.colorbar(scatter, label=r"Eucledian distance: $\|\mathbf{I} - \mathbf{\hat{I}}\|$")
    plt.savefig("./figs/reconstruc_error_arena.svg")
    plt.show()
    
    plt.figure(figsize=(5,5))
    plt.hist(all_errors, bins=20)
    plt.title("Reconstruction error distribution")
    plt.xlabel(r"Eucledian distance: $\|\mathbf{I} - \mathbf{\hat{I}}\|$")
    plt.savefig("./figs/image_ceconstr_distr.svg")
    plt.show()


def check_z_distribution(device, model, test_loader, zfromH=False, save_z=False):
    all_embeddings = []
    all_positions = []
    model.eval()
    
    # print(len(test_loader))
    # for i, (x_batch, pos_batch) in enumerate(train_loader):
    for i, (x_batch, pos_batch) in enumerate(test_loader):
    # for x_batch, pos_batch in test_loader:
        x_batch = x_batch.to(device)
        pos_batch = pos_batch.to(device)
        
        with torch.no_grad():
            if not zfromH:
                z = model.encode(x_batch)
            else:
                z = model.position2latent_mapping(pos_batch)
        all_embeddings.append(z.cpu())  
        all_positions.append(pos_batch.cpu())
    
    
    all_positions = torch.concat(all_positions).numpy()
    all_embeddings = torch.concat(all_embeddings).numpy()
    print(all_positions.shape)
    print(all_embeddings.shape)
    if save_z:
        np.save('positions.npy', all_positions)
        np.save('z_embeddings.npy', all_embeddings)
    
    mu_mle, sigma_mle = mle(all_embeddings)  
    im = plt.imshow(sigma_mle)
    plt.title(rf"Covariance matrix $\Sigma$ zfromH: {zfromH}")
    plt.colorbar(im)
    plt.show()
    plt.hist(all_embeddings[:, 10])
    plt.show()
    print(mu_mle, sigma_mle)
    return mu_mle, sigma_mle
        

def plot_z_position_correlation(device, model, test_loader):
    all_embeddings = []
    all_positions = []
    # all_embeddings_pred = []
    
    H = model.position2latent_mapping.weight.data.cpu().numpy()
    # print(len(test_loader))
    # for i, (x_batch, pos_batch) in enumerate(train_loader):
    for i, (x_batch, y_batch) in enumerate(test_loader):
    # for x_batch, pos_batch in test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        model.train()
        with torch.no_grad():
            z = model.encode(x_batch)
            # print(z)
            # all_embeddings.append(z.cpu())  
            all_embeddings.append( model.position2latent_mapping(y_batch).cpu())
            all_positions.append(y_batch.cpu())
        
        if i>10:
            break
    all_positions = torch.concat(all_positions).numpy()
    all_embeddings = torch.concat(all_embeddings).numpy()
    print(all_positions.shape)
    print(all_embeddings.shape)
    
    # out = pearsonr(all_embeddings, all_positions)
    # print(out)
    
    # Initialize a matrix to store the correlation coefficients
    correlations = np.zeros((all_positions.shape[1], all_embeddings.shape[1]))

    # Calculate correlations for each dimension
    for i in range(all_positions.shape[1]):  # Iterate over each position dimension
        for j in range(all_embeddings.shape[1]):  # Iterate over each embedding dimension
            corr, _ = pearsonr(all_positions[:, i], all_embeddings[:, j])
            correlations[i, j] = corr
    
    plt.figure(figsize=(3,3))
    plt.scatter([0]*24, correlations[0], s=5, label=r"X correlation, each $m_{i}$ dim")
    plt.scatter([1]*24, correlations[1], s=5, label=r"Y correlation, each $m_{i}$ dim")
    plt.xlim(-1,2)
    plt.ylim(-1,1)
    plt.legend()
    plt.xticks([])
    plt.show()
    
def plot_logz_dist_arena(device, model, test_loader, zfromH=False, plot_z_dist=False):
    all_embeddings = []
    all_positions = []
    all_embeddings_pred = []
    
    if not plot_z_dist:
        mu_mle, sigma_mle = check_z_distribution(device, model, test_loader, zfromH=zfromH)
        
    
    # H = model.position2latent_mapping.weight.data.cpu().numpy()
    # print(len(test_loader))
    # for i, (x_batch, pos_batch) in enumerate(train_loader):
    for i, (x_batch, y_batch) in enumerate(test_loader):
    # for x_batch, pos_batch in test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        model.eval()
        with torch.no_grad():
            z = model.encode(x_batch)
            # print(z)
            all_embeddings.append(z.cpu())  
            all_positions.append(y_batch.cpu())
            all_embeddings_pred.append( model.position2latent_mapping(y_batch).cpu())
        
        if i>6:
            break
    
    all_positions = torch.concat(all_positions).numpy()
    all_embeddings = torch.concat(all_embeddings).numpy()
    all_embeddings_pred = torch.concat(all_embeddings_pred)
    all_metrics = []
    fig, ax = create_arena_canvas()
    print(all_embeddings)
    print(all_embeddings_pred)
    for i in range(len(all_embeddings)):
        if not plot_z_dist:
            nll = nlog_likelihood(all_embeddings[i], mu_mle, sigma_mle)
            metric = nll
        else: 
            mse = np.mean((all_embeddings[i] - all_embeddings_pred[i].numpy())**2)
            metric = mse
        # sc = ax.scatter(all_positions[i][0]*55, all_positions[i][1]*55, c=metric, cmap='viridis', vmin=0, vmax=.2, alpha=.8)
        all_metrics.append(metric)
    
    min_ = (min(all_metrics))
    max_ = (max(all_metrics))
    print(min_, max_)
    sc = ax.scatter(all_positions[:,0]*55, all_positions[:,1]*55, c=all_metrics, cmap='viridis', vmin=min_, vmax=max_, alpha=.8)
    if plot_z_dist:
        plt.title("MSE between z $\hat{z}$ per position")
        plt.colorbar(sc, label="MSE")
    else:
        plt.title(r"Negative log likelihood $z_{i}$ per position")
        plt.colorbar(sc, label="Negative log likelihood")
    
    plt.savefig(f"./figs/plot_logz_dist_arena.svg")
    plt.show()

    # all_nlls = []
    # fig, ax = create_arena_canvas()
    # for i in range(len(all_embeddings)):
    #     nll = nlog_likelihood(all_embeddings_pred[i], mu_mle, sigma_mle)
    #     sc = ax.scatter(all_positions[i][0]*55, all_positions[i][1]*55, c=nll, cmap='viridis', vmin=611, vmax=730, alpha=.8)
    #     all_nlls.append(nll)
    # print(min(all_nlls))
    # print(max(all_nlls))
    # plt.colorbar(sc, label="Negative log likelihood")
    # plt.title("Negative log likelihood $z_{i}$ per position H mapping")
    # plt.show()
    
    
    # plt.savefig("./figs/nll_z_arena.svg")
    
def check_H_mapping(device, model, test_loader):
    all_embeddings = []
    all_positions = []
    # print(len(test_loader))
    # for i, (x_batch, pos_batch) in enumerate(train_loader):
    for i, (x_batch, pos_batch) in enumerate(test_loader):
        # legit_angle_idx = torch.where((pos_batch[:, 2] > 85) & (pos_batch[:, 2] <95))[0]
        # if len(legit_angle_idx) == 0:
        #     continue
        
        # x_batch = x_batch[legit_angle_idx]
        # pos_batch = pos_batch[legit_angle_idx]

        x_batch = x_batch.to(device)
        pos_batch = pos_batch.to(device)
        
        if i<3:
            model.train()
        else:
            model.eval()
        with torch.no_grad():
            z_batch = model.encode(x_batch)
            
        x_reconstr_batch = model.decode(z_batch)
        
        z_batch_pred = model.position2latent_mapping(pos_batch)
        print(model.position2latent_mapping.bias.data)
        print(z_batch_pred[0])
        print(model.position2latent_mapping.weight.data@pos_batch[0] +model.position2latent_mapping.bias.data)
        x_reconstr_batch_pred = model.decode(z_batch_pred)
        
        fig, ax = plt.subplots(1,3)
        
        ax[0].imshow(x_batch[0].detach().cpu().numpy().squeeze(), cmap='gray', vmin=0, vmax=1)
        ax[0].set_title("$I$")
        ax[0].axis('off')
        
        ax[1].imshow(x_reconstr_batch[0].detach().cpu().numpy().squeeze(), cmap='gray', vmin=0, vmax=1)
        ax[1].set_title(r"$f_{\theta}(h_{\phi}(I_{i})) = \hat{I_{i}}$")
        ax[1].axis('off')
    
        ax[2].imshow(x_reconstr_batch_pred[0].detach().cpu().numpy().squeeze(), cmap='gray', vmin=0, vmax=1)
        ax[2].set_title(r"$f_{decoder}(p^{T}H) = \hat{I_{i}}$")
        ax[2].axis('off')
        
        plt.savefig(f"./figs/pos2z_reconstruction_{i}.svg")
        plt.show()
        
        
        print(z_batch.shape)
        print(pos_batch.shape)
        
        if i>6:
            break
        

def plt_4z_dims(Z):
    dims = (2,11,13,21)
    dims = (7,13,14,19)
    fig, ax = plt.subplots(4, 4, figsize=(10, 10), sharex=True)#, sharey=True)

    # turn off y ticks and ytick labels for all axis
    for a in ax.flatten():
        a.yaxis.tick_right()
    
        
    
    for d in dims:
        for with_dim in dims:
            axis = ax[dims.index(d), dims.index(with_dim)]
            if dims.index(d) == dims.index(with_dim):
                axis.hist(Z[:, d], edgecolor='white', bins=20, linewidth=.4, color='gray')
                axis.set_title(fr"$m_{{{d}}}$")
            else:
                axis.scatter(Z[:, d], Z[:, with_dim], c='k', s=1, alpha=.1)
                axis.set_title(fr"$m_{{{d}}}, m_{{{with_dim}}}$")
                axis.spines['top'].set_visible(False)
                axis.spines['left'].set_visible(False)

    plt.savefig(f"./figs/4dims_distr.svg")
    plt.show()
    

def main():
    latent_dims = 24 
    batch_size = 256
    device = get_device()
    
    path = '/Users/loaloa/homedataAir/phd/ratvr/VirtualReality/data/2024-07-14_21-37_mlDataset90onlyUniform'
    # path = '/Users/loaloa/homedataAir/phd/ratvr/VirtualReality/data/2024-07-16_14-21_mlDataset90onlyUniformeSmallwall'
    # path = '/Users/loaloa/homedataAir/phd/ratvr/VirtualReality/data/2024-07-16_14-16_mlDataset90onlyUniformWihtewall'
    train_loader, test_loader = get_dataloaders(path, batch_size)

    model = VAE(latent_dims).to(device)
    model.load_state_dict(torch.load('./vaelulGAMMA_E3.pth'))
    # model.load_state_dict(torch.load('./vae_finalKL.3_E5.pth'))
    
    Z = np.load("./z_embeddings.npy")
    # positions = np.load("./positions.npy")
    
    # print(nlog_likelihood(z[0], mu_mle, sigma_mle))
    # nlog_likelihood(z[1], mu_mle, sigma_mle)
    
    plot_latent_dims(device, model, train_loader, test_loader, zfromH=False)
    plot_latent_dims(device, model, train_loader, test_loader, zfromH=True)
    plot_reconstruction_error(device, model, train_loader, test_loader)
    plot_pos_errors(device, model, test_loader, zfromH=False  )
    plot_pos_errors(device, model, test_loader, zfromH=True  )
    # mu_mle, sigma_mle = check_z_distribution(device, model, test_loader, zfromH=False, save_z=True)
    # mu_mle, sigma_mle = check_z_distribution(device, model, test_loader, zfromH=True, save_z=True)
    
    plot_logz_dist_arena(device, model, test_loader, zfromH=False,  plot_z_dist=False)
    
    plt_4z_dims(Z)
    
    # check_H_mapping(device, model, test_loader)
    # plot_z_position_correlation(device, model, test_loader)
    
if __name__ == "__main__":
    main()