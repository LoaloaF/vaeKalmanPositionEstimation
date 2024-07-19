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
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import warnings
# from catalyst.utils import set_global_seed
from copy import deepcopy
from sklearn.preprocessing import QuantileTransformer
import plotly.express as px
sns.set_style('darkgrid')
warnings.filterwarnings("ignore")


np.random.seed(42)
torch.manual_seed(42)
from VAE import VAE
from UnityLocalizationDataset import UnityLocalizationDataset
from evaulate_model import check_H_mapping
from evaulate_model import plot_z_position_correlation

from evaulate_model import plot_latent_dims
from evaulate_model import plot_pos_errors
from evaulate_model import plot_logz_dist_arena
from evaulate_model import check_z_distribution
from evaulate_model import plt_4z_dims

def get_device():
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print(x)
        return mps_device
    elif torch.cuda.is_available():
        x = torch.ones(1, device=torch.device("cuda"))
        print(x)
        return torch.device("cuda")
    else:
        print ("Nor MPS or cuda device found.")
        exit(1)
        
def get_dataloaders(path, batch_size=128):
    train_dataset = UnityLocalizationDataset(path, proportion=0.9, train=True)
    test_dataset = UnityLocalizationDataset(path, proportion=0.1, train=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=batch_size, 
                                              shuffle=False)
    return train_loader, test_loader
    
            
def KL_divergence(mu, logsigma):
    loss = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
    return loss

# def log_likelihood(x, reconstruction):
#     loss = nn.BCELoss(reduction='sum')
#     return loss(reconstruction, x)

def log_likelihood(x, reconstruction, collapse_batch_dim=True):
    if collapse_batch_dim:
        loss = nn.MSELoss(reduction='mean')
    else:
        return nn.MSELoss(reduction='none')(reconstruction, x).sum(axis=(1,2,3))
    return loss(reconstruction, x)*1e4

def loss_vae(x, mu, logsigma, reconstruction, alpha, beta, verbose=True):
    # print(f"With x={x.shape}, mu={mu.shape}, logsigma={logsigma.shape}, reconstruction={reconstruction.shape}")
    KL_loss_comp = alpha*(KL_divergence(mu, logsigma))
    log_likelihood_comp = beta*(log_likelihood(x, reconstruction))*5
    if verbose:
        print(f"KL_loss_comp={int(KL_loss_comp):,<6}, log_likelihood_comp={int(log_likelihood_comp):,<6}", end=" ")
    return KL_loss_comp, log_likelihood_comp

def cosine_distance_between_angles(angle1, angle2):
    # Convert angles from degrees to radians
    angle1_rad = angle1 * torch.pi / 180
    angle2_rad = angle2 * torch.pi / 180
    # Convert angles to points on the unit circle
    x1, y1 = torch.cos(angle1_rad), torch.sin(angle1_rad)
    x2, y2 = torch.cos(angle2_rad), torch.sin(angle2_rad)
    # Calculate cosine similarity
    cosine_similarity = (x1 * x2 + y1 * y2) / (torch.sqrt(x1**2 + y1**2) * torch.sqrt(x2**2 + y2**2))
    # Calculate cosine distance
    cosine_distance = torch.sum(1 - cosine_similarity) *10000
    # print(cosine_similarity)
    return cosine_distance

def train_epoch(device, model, criterion, optimizer, data_loader, alpha, beta, gamma):
    train_losses_per_epoch = []
    model.train()
    
    for i, (x_batch, y_batch) in enumerate(data_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # seperate loss case        
        # mu, logsigma, reconstruction = model(x_batch)
        # loss1, loss2 = criterion(x_batch.to(device).float(), mu, logsigma, reconstruction, alpha, beta, gamma)
        # z = model.encode(x_batch)
        # z_pred = model.position2latent_mapping(y_batch)
        # loss3 = (1-(alpha+beta)) * nn.MSELoss(reduction='mean')(z, z_pred)*150
        # print(f"z loss={loss3:,<6}")
        
        mu, logsigma, reconstruction = model(x_batch)
        loss1, loss2 = criterion(x_batch.float(), mu, logsigma, reconstruction, alpha, beta)
        
        mu, logsigma, reconstruction, z, z_pred = model.zmap_forward_pass(x_batch, y_batch)
        # loss1_, loss2_ = criterion(x_batch.float(), mu, logsigma, reconstruction, alpha, beta, verbose=False)
        # loss2 = (loss2 + loss2_)/2
        # loss1 = (loss1 + loss1_)/2
        loss3 = gamma * nn.MSELoss(reduction='mean')(z, z_pred)*150
        print(f"NLL z loss={loss3:.2f}", end="  ")
        
        pos = model.latent2position_mapping(z_pred)
        loss4 = (1-(alpha+beta+gamma)) * nn.MSELoss()(pos, y_batch)*1000
        
        print(f"z2postition loss={loss4:.2f}")
        

        optimizer.zero_grad()
        (loss1+loss2+loss3+loss4).backward()
        optimizer.step()

        train_losses_per_epoch.append(np.array([loss1.item(), loss2.item(), loss3.item(), loss4.item()]))
        # if i>100:
        #     break

    return train_losses_per_epoch, mu, logsigma, reconstruction

def eval_epoch(device, model, criterion, optimizer, data_loader, alpha, beta, gamma):
    val_losses_per_epoch = []
    model.eval()
    with torch.no_grad():
        for i, (x_val, y_batch) in enumerate(data_loader):
            x_val = x_val.to(device)
            y_batch = y_batch.to(device)
            mu, logsigma, reconstruction = model(x_val)
            loss1, loss2 = criterion(x_val.to(device).float(), mu, logsigma, reconstruction,alpha, beta)
            z = model.encode(x_val)
            z_pred = model.position2latent_mapping(y_batch)
            loss3 = nn.MSELoss(reduction='mean')(z, z_pred)
            val_losses_per_epoch.append(np.array([loss1.item(), loss2.item(), loss3.item()]))
            
            # if i  >100:
            #     break
    return val_losses_per_epoch, mu, logsigma, reconstruction

def plot_output(device, model, epoch, epochs, train_loader, test_loader, 
                train_loss=None, val_loss=None, size = 5):
    clear_output(wait=True)
    plt.figure(figsize=(18, 6))
    
    # test_dataset = datasets.MNIST(root='./data/', train=False, 
    #                                transform=transforms.ToTensor(), 
    #                                download=False, )
    path = '/Users/loaloa/homedataAir/phd/ratvr/VirtualReality/data/2024-07-14_21-37_mlDataset90onlyUniform'
    test_dataset = UnityLocalizationDataset(path, proportion=0.2, train=False)

    for k in range(size):
        ax = plt.subplot(2, size, k + 1)
        img = test_dataset[k][0].unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
             mu, logsigma, reconstruction  = model(img)

        plt.imshow(img.cpu().squeeze().numpy(), cmap='gray')
        plt.axis('off')
        if k == size//2:
            ax.set_title('Real')
        ax = plt.subplot(2, size, k + 1 + size)
        plt.imshow(reconstruction.cpu().squeeze().numpy(), cmap='gray')
        plt.axis('off')

        if k == size//2:
            ax.set_title('Output')
    if train_loss is not None and val_loss is not None:
        plt.suptitle('%d / %d - loss: %f val_loss: %f' % (epoch+1, epochs, train_loss, val_loss))
    plt.show()
    plt.savefig(f'./results/{epoch}.png')

def model_eval_plots(device, model, train_loader, test_loader):
    plot_latent_dims(device, model, train_loader, test_loader, zfromH=False)
    # plot_latent_dims(device, model, train_loader, test_loader, zfromH=True)
    # plot_pos_errors(device, model, test_loader, zfromH=False)
    # plot_pos_errors(device, model, test_loader, zfromH=True)
    
    
    plot_logz_dist_arena(device, model, test_loader, zfromH=False,  plot_z_dist=False)
    # plot_logz_dist_arena(device, model, test_loader, zfromH=False,  plot_z_dist=True)
    # plot_logz_dist_arena(device, model, test_loader, zfromH=True,  plot_z_dist=False)
    # plot_logz_dist_arena(device, model, test_loader, zfromH=True,  plot_z_dist=True)
    
    mu_mle, sigma_mle = check_z_distribution(device, model, test_loader, zfromH=False, save_z=True)
    Z = np.load("./z_embeddings.npy")
    plt_4z_dims(Z)

    
    check_H_mapping(device, model, test_loader)
    
def train_loop(device, model, loss_fn, optimizer, train_loader, test_loader, epochs):
    loss = {'train_loss':[],'val_loss':[]}
    train_loss = None
    val_loss = None
    
    # alpha = 0 # KL wesight
    # beta = 0 # NLL weight
    # gamma = 0 
    alpha = .25 # KL wesight
    beta = .75 # NLL weight
    gamma = 0 # zmap weight, H
    # rest is z2position weight, sums to 1
    
    with tqdm(desc="epoch", total=epochs) as pbar_outer:
        for epoch in range(epochs):
            print('* Epoch %d/%d' % (epoch+1, epochs))
            # model_eval_plots(device, model, train_loader, test_loader)
            
            train_loss, mu, logsigma, reconstruction = (train_epoch(device, model, 
                                                                loss_fn, 
                                                                optimizer, 
                                                                train_loader,
                                                                alpha, beta, gamma,
                                                                )
            )

            val_loss, mu, logsigma, reconstruction = (eval_epoch(device, model, 
                                                            loss_fn, 
                                                            optimizer, 
                                                            test_loader,
                                                            alpha, beta, gamma,
                                                            )
            )
            train_loss = np.stack(train_loss)
            val_loss = np.stack(val_loss)
            print("Train loss: ", train_loss, "Val loss: ", val_loss)
            pbar_outer.update(1)

            loss['train_loss'].append(train_loss.mean())
            loss['val_loss'].append(val_loss.mean())
            torch.save(model.state_dict(), f'./vae_finalKL.3_E{epoch+1}.pth')
            
            plot_loss((train_loss, val_loss))
            model_eval_plots(device, model, train_loader, test_loader)
            # check_H_mapping(device, model, test_loader)
            # plot_z_position_correlation(device, model, test_loader)
            # plot_pos_errors(device, model, None, test_loader)
            # plot_output(device, model, epoch, epochs, train_loader, test_loader, train_loss.mean(), val_loss.mean(), size = 10)
            
            # if input("Continue? (y/n)") == "n":
            #     break
            alpha = float(input(f"Enter alpha (cur value: {alpha}): "))
            beta = float(input(f"Enter beta (cur value: {beta}): "))
    model_eval_plots(device, model, train_loader, test_loader)
    
    # plot_loss((train_loss, val_loss))
    # check_H_mapping(device, model, test_loader)
    # plot_output(device, model, epoch, epochs, train_loader, test_loader, train_loss.mean(), val_loss.mean(), size = 10)
    plot_loss(loss)

# def train_loop2(device, model, loss_fn, optimizer, train_loader, test_loader, epochs):
#     def train_epoch(device, model, criterion, optimizer, data_loader):
#         train_losses_per_epoch = []
#         model.train()
#         for x_batch, y_batch in data_loader:
#             x_batch = x_batch.to(device)
#             y_batch = y_batch.to(device)
#             print(y_batch[:, :2].min(), y_batch[:, :2].max())
#             z = model.encode(x_batch)
            
#             y_pred = model.latent2position_mapping(z)
#             loss_pos, loss_angle = loss_position_mapping(y_pred, y_batch)
#             loss = loss_pos+loss_angle
#             print(f"pos_loss_comp={loss_pos:,<6}, consine_loss_comp={loss_angle.item():,<6}")
            

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             train_losses_per_epoch.append(np.array((loss_pos.detach().item(), loss_angle.detach().item())))
        
#         train_losses_per_epoch = np.array(train_losses_per_epoch)
#         plt.plot(train_losses_per_epoch[:,0])
#         plt.plot(train_losses_per_epoch[:,1])
#         plt.show()
#         return np.mean(train_losses_per_epoch)
    
def train_loop3(device, model, loss_fn, optimizer, train_loader, test_loader, epochs):
    def train_epoch(device, model, criterion, optimizer, data_loader):
        train_losses_per_epoch = []
        model.train()
        for i, (x_batch, y_batch) in enumerate(data_loader):
            legit_angle_idx = torch.where((y_batch[:, 2] > 85) & (y_batch[:, 2] <95))[0]
            if len(legit_angle_idx) == 0:
                continue
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            z = model.encode(x_batch)
            z_pred = model.position2latent_mapping(y_batch)
            loss = nn.MSELoss(reduction='mean')(z, z_pred)
            print(f"z loss={loss:,<6}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses_per_epoch.append(loss.detach().item())
        
            # if i >2:
            #     break
        
        plt.plot(train_losses_per_epoch)
        plt.show()

        return train_losses_per_epoch
    
    # train only the position2latent_mapping
    model.position2latent_mapping.bias = nn.Parameter(torch.zeros_like(model.position2latent_mapping.bias, dtype=torch.float32))
    for param in model.parameters():
        param.requires_grad = False
    model.position2latent_mapping.weight.requires_grad = True    
    
    loss = {'train_loss':[],'val_loss':[]}
    with tqdm(desc="epoch", total=epochs) as pbar_outer:
        for epoch in range(epochs):
            print('* Epoch %d/%d' % (epoch+1, epochs))
            train_loss = (train_epoch(device, model, 
                                      loss_fn, 
                                      optimizer, 
                                      train_loader
                                      )
            )
            loss['train_loss'].extend(train_loss)

            check_H_mapping(device, model, test_loader)
            torch.save(model.state_dict(), f'./vae_model_weights_BigLMarks_E{epoch}_ZWeights7.pth')
            if input("Continue? (y/n)") == "n":
                break
        plot_loss(loss)
            
            

def plot_loss(loss):
    plt.figure(figsize=(15, 6))
    if isinstance(loss, tuple):
        print(loss[0].shape)
        plt.plot(loss[0][:,0], label='Train KL', color='blue')
        plt.plot(loss[0][:,1], label='Train NLL', color='blue', alpha=0.5)
        plt.plot(loss[0][:,2], label='Train zdist MSE', color='purple', alpha=0.2)
        plt.plot(loss[0][:,3], label='Train pos2z MSE', color='blue', alpha=0.2)
        plt.plot(loss[1][:,0], label='Valid KL', color='red')
        plt.plot(loss[1][:,1], label='Valid NLL', color='red', alpha=0.5)
        plt.plot(loss[1][:,2], label='Valid pos2z MSE', color='red', alpha=0.2)
    else:
        plt.semilogy(loss['train_loss'], label='Train')
        plt.semilogy(loss['val_loss'], label='Valid')
        plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.title('Loss_vae')
    plt.show()

def main():
    latent_dims = 24
    learning_rate = 1e-3
    epochs = 3
    batch_size = 256
    
    device = get_device()
    
    path = '/Users/loaloa/homedataAir/phd/ratvr/VirtualReality/data/2024-07-08_22-24_mlDatasetBigLmarks3'
    path = '/Users/loaloa/homedataAir/phd/ratvr/VirtualReality/data/2024-07-14_21-37_mlDataset90onlyUniform'
    train_loader, test_loader = get_dataloaders(path, batch_size)
    
    # train_loader, test_loader = get_MNIST_dataloaders(batch_size)
    
    autoencoder_vae = VAE(latent_dims).to(device)
    # plot_output(device, autoencoder_vae, -1, epochs, train_loader, test_loader, None, None, size = 10)
    
    autoencoder_vae.load_state_dict(torch.load('./vae_finalKL.3_E5.pth'))
    # autoencoder_vae.position2latent_mapping.weight.requires_grad = True
    # autoencoder_vae.position2latent_mapping.weight = torch.nn.Parameter(torch.randn(latent_dims, 3, device=device))
    
    # # Load the saved state dictionary
    # saved_state_dict = torch.load('./vae_model_weights_BigLMarks_E10.pth')
    # # Get the current model's state dictionary
    # current_state_dict = autoencoder_vae.state_dict()
    # # Update the current state dictionary with the saved state dictionary
    # # This will only update the keys that exist in both dictionaries, ignoring the rest
    # updated_state_dict = {k: v for k, v in saved_state_dict.items() if k in current_state_dict}
    # # Load the updated state dictionary back into the model
    # autoencoder_vae.load_state_dict(updated_state_dict, strict=False)

    
    
    optimizer = torch.optim.Adam(autoencoder_vae.parameters(), lr=learning_rate)
    train_loop(device, autoencoder_vae, loss_vae, optimizer, train_loader, test_loader, 
               epochs=epochs)
    # learning_rate = 1e-2
    # epochs = 6
    # train_loop3(device, autoencoder_vae, loss_vae, optimizer, train_loader, test_loader, 
    #            epochs=epochs)
    
    # save the weights of the model
    # torch.save(autoencoder_vae.state_dict(), f'./vae_model_weights_BigLMarks_E5_ZWeights6.pth')
    
if __name__ == "__main__":
    main() 