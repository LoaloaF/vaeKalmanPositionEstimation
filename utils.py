import torch
import torch.nn as nn

from UnityLocalizationDataset import UnityLocalizationDataset
import matplotlib.pyplot as plt

import numpy as np

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
    
def create_arena_canvas():
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca()
    [ax.spines[spine].set_visible(False) for spine in ["top", "right", "bottom", "left"]]
    plt.vlines([-50,50], -55,55, edgecolor='k', linestyles='--', alpha=.3)
    plt.hlines([-50,50], -55,55, edgecolor='k', linestyles='--', alpha=.3)
    plt.xlim(-55,55)
    plt.ylim(-55,55)
    
    plt.grid(alpha=.3)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor("#eaeaf2ff")
    return fig, ax

def KL_divergence_general(mu1, sigma1, mu2, sigma2):
    """
    Calculate the KL divergence between two normal distributions.
    
    Parameters:
    - mu1, sigma1: Mean and standard deviation of the first distribution.
    - mu2, sigma2: Mean and standard deviation of the second distribution.
    
    Returns:
    - The KL divergence between the two distributions.
    """
    sigma1_squared = sigma1**2
    sigma2_squared = sigma2**2
    
    loss = np.log(sigma2/sigma1) + (sigma1_squared + (mu1 - mu2)**2) / (2 * sigma2_squared) - 0.5
    return np.sum(loss)

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
        ax[1].set_title(r"$f_{decoder}(f_{encoder}(I_{i})) = \hat{I_{i}}$")
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
        
def log_likelihood(x, reconstruction, collapse_batch_dim=True):
    if collapse_batch_dim:
        loss = nn.MSELoss(reduction='mean')
    else:
        return nn.MSELoss(reduction='none')(reconstruction, x).sum(axis=(1,2,3))
    return loss(reconstruction, x)*1e4

