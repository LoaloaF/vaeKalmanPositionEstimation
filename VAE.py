import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):

    def __init__(self, dim_code):
        super().__init__()
        # encoder

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = 8, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size = 8, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size = 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()

        )
        
        latent_dim_img = 7 # 28 / 2 / 2 MNIST
        latent_dim_img = 3 # 224 / 2 / 2 Unity data

        self.flatten_mu = nn.Linear(128 * latent_dim_img*latent_dim_img, out_features=dim_code)
        self.flatten_logsigma = nn.Linear(128 * latent_dim_img*latent_dim_img, out_features=dim_code)

        # decoder
        self.decode_linear = nn.Linear(dim_code, 128 * latent_dim_img*latent_dim_img)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size = 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size = 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size = 8, stride=4, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size = 8, stride=4, padding=1),
            nn.Sigmoid(),
            nn.ConstantPad2d((11,11,11,11), 0.5),
        )
        
        self.latent2position_mapping = nn.Linear(dim_code, 3)
        self.latent2position_mapping.bias = nn.Parameter(torch.zeros_like(self.latent2position_mapping.bias, dtype=torch.float32))
        self.latent2position_mapping.bias.requires_grad = False
        self.position2latent_mapping = nn.Linear(3, dim_code)
        self.position2latent_mapping.bias = nn.Parameter(torch.zeros_like(self.position2latent_mapping.bias, dtype=torch.float32))
        self.position2latent_mapping.bias.requires_grad = False
        
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu, logsigma = self.flatten_mu(x), self.flatten_logsigma(x)
        z = self.gaussian_sampler(mu, logsigma)
        return z
    
    def gaussian_sampler(self, mu, logsigma):
        if self.training:
            std = torch.exp(logsigma / 2)
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


    def decode(self, x):
        latent_dim_img = 7 # 28 / 2 / 2 MNIST
        latent_dim_img = 3 # 224 / 2 / 2 Unity data
        
        x = self.decode_linear(x)
        x = x.view(x.size(0), 128, latent_dim_img, latent_dim_img)
        # x = F.relu(self.decode_2(x))
        # reconstruction = F.sigmoid(self.decoder(x))
        reconstruction = self.decoder(x)
        return reconstruction
    
    def forward(self, x):
        latent_dim_img = 7 # 28 / 2 / 2 MNIST
        latent_dim_img = 3 # 224 / 2 / 2 Unity data
        
        x = self.encoder(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        # print(x.size())
        mu, logsigma = self.flatten_mu(x), self.flatten_logsigma(x)
        z = self.gaussian_sampler(mu, logsigma)
        x = self.decode_linear(z)
        x = x.view(x.size(0), 128, latent_dim_img, latent_dim_img)
        # x = F.relu(self.decode_2(x))
        # reconstruction = F.sigmoid(self.decode_1(x))
        reconstruction = self.decoder(x)
        return mu, logsigma, reconstruction
    
    def zmap_forward_pass(self, x, y):
        latent_dim_img = 3 # 224 / 2 / 2 Unity data
        x = self.encoder(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        # print(x.size())
        mu, logsigma = self.flatten_mu(x), self.flatten_logsigma(x)
        z = self.gaussian_sampler(mu, logsigma)
        
        # use the z predicted by y (position) to reconstruct the image (harder)
        z_pred = self.position2latent_mapping(y)
        
        x = self.decode_linear(z_pred)
        x = x.view(x.size(0), 128, latent_dim_img, latent_dim_img)
        # x = F.relu(self.decode_2(x))
        # reconstruction = F.sigmoid(self.decode_1(x))
        reconstruction = self.decoder(x)
        return mu, logsigma, reconstruction, z, z_pred
        