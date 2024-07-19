import numpy as np
import requests
import time
from acquire_dataset import initate, start_unity, start_session, term_delete
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

import sys
sys.path.append("../CoreRatVR")
sys.path.append("../CoreRatVR/SHM")
from CyclicPackagesSHMInterface import CyclicPackagesSHMInterface
from VideoFrameSHMInterface import VideoFrameSHMInterface

base_url = "http://localhost:8000"

import asyncio
import websockets
import json

import matplotlib.pyplot as plt

import numpy.linalg as linalg

import torch
from VAE import VAE
from utils import get_device
from z_prob_estimation import mle, nlog_likelihood

from utils import create_arena_canvas
import cv2

from utils import KL_divergence_general


def sample_location():
    x, y = np.random.multivariate_normal([0, 0], [[400, 0], [0, 400]], 1)[0]
    x, y = np.random.uniform(-50, 50, 2)
    x = min(max(-50, x), 50)
    y = min(max(-50, y), 50)
    return np.array([x, y])

def draw_uncertainty(ax, mean, cov, edgecolor='None', fc='r', label=None):
    # Calculate eigenvalues and eigenvectors for the covariance matrix
    mean = mean[:2]  # Only consider the 2D mean
    # print("plot:", cov.shape, cov, mean)
    cov = cov[:2, :2]  # Only consider the 2D covariance
    vals, vecs = linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    
    # Calculate the angle of rotation and dimensions of the ellipse
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(vals)  # 2*sqrt(eigenvalues) gives the length of the ellipse axes
    
    # Create an ellipse patch
    one_std = patches.Ellipse(xy=mean, width=width, height=height, angle=theta, 
                              edgecolor=edgecolor, fc=fc, lw=2, alpha=.1)
    two_std = patches.Ellipse(xy=mean, width=width*2, height=height*2, angle=theta, 
                              edgecolor=edgecolor, fc=fc, lw=2, alpha=.1)
    ax.scatter(mean[0], mean[1], c=edgecolor, s=4, alpha=1)
    ax.text(mean[0], mean[1], label, fontsize=8, c='k', alpha=.5)
    # ax.add_patch(one_std)
    ax.add_patch(two_std)

def get_next_frame(unityframe_shm):
    frame = unityframe_shm.get_frame()
    # rotate image 180 degrees
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize the frame to a fixed size (e.g., 224x224)
    frame = cv2.resize(frame, (224, 224))
    # Convert frame to float32, normalize to range [0, 1]
    frame = torch.tensor(frame, dtype=torch.float32) / 255.0
    return frame.unsqueeze(0)

        
def kalman_filter(vae, mu_mle, sigma_mle, device):
    seed = 3
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    
    vel_step_size = 5
    # set the covariance/uncertainty about vae and measument estimates
    Q = np.array([[3, 0, 0],
                  [0, 3, 0],
                  [0, 0, 1]])
    R = np.eye(24)
    # transition vae matrix
    H = vae.position2latent_mapping.weight.data.cpu().numpy()
    # initialize the state, original previous state estimate
    s_prv_given_m_prv_mean = np.array([0, 0, 0])
    s_prv_given_m_prv_cov = np.array([[1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1]])
    
    # initialize the VAE vae for encoding images to latent var m
    vae.to(device)
    # vae.train()
    vae.eval()
    # setup shared memory
    unityout_shm = CyclicPackagesSHMInterface("../tmp_shm_structure_JSONs/unityoutput_shmstruct.json")
    unityframe_shm = VideoFrameSHMInterface("../tmp_shm_structure_JSONs/unitycam_shmstruct.json")
    unityout_shm.reset_reader()
    
    # fig, ax = create_arena_canvas()

    goal_xy = (0,0)
    i = 0
    requests.post(f"{base_url}/unityinput/Teleport%2C{0:.3f}%2C{0:.3f}%2C{90:.3f}")
    time.sleep(.2)
    val1_all, val2_all, val3_all = [], [], []
    while True:
        # check if a new frame was rendered
        if not (unityout_shm.usage > 0):
            continue
        
        # get the corrent location for plotting the true trajectory
        pack = unityout_shm.popitem(return_type=dict)
        x_cur_true, y_cur_true, alpha = pack["X"], pack["Z"], pack["A"]
        print("True: ", x_cur_true, y_cur_true)
        if i == 0:
            x_prv_true, y_prv_true = x_cur_true, y_cur_true
            

        # check if goal reached
        if np.linalg.norm(goal_xy - np.array([x_cur_true, y_cur_true])) <10: #and i >200:
            print(i)
            print("===============================")
            plt.show()
            plt.close()

            fig, ax = create_arena_canvas()
            fig2, ax2 = plt.subplots(nrows=3, ncols=1, figsize=(6, 3), sharex=True)
            
            ylims = ((15,18),(2.5,5.5),(-1e2,1e6))
            titles = (r"$\hat{m_{i}}$ NLL", r"H error $\|m_{i} - \hat{m_{i}}\|$", r"$\mathbf{KL}(p(s_{i}|m_{i-1}) \| p(s_{i}|m_{i})$)")
            for i in range(3):
                ax2[i].set_title(titles[i], pad=-5)
                ax2[i].set_xlabel("step i")
                # ax2[i].set_ylim(ylims[i])
                ax2[i].spines["top"].set_visible(False)
                ax2[i].spines["right"].set_visible(False)
                ax2[i].spines["bottom"].set_visible(False)

            # reset the state
            goal_xy = sample_location()
            ax.scatter(goal_xy[0], goal_xy[1], marker='x', c="k", s=6, alpha=.8)
            s_prv_given_m_prv_mean = np.array([x_cur_true, y_cur_true, 0])
            s_prv_given_m_prv_cov = np.array([[1, 0, 0],
                                              [0, 1, 0],
                                              [0, 0, 0]])
            i = 0
            draw_uncertainty(ax, s_prv_given_m_prv_mean, s_prv_given_m_prv_cov, fc='yellow', label=i)
            


        # calculate the movement direction wrt the goal
        goal_vector = ((goal_xy - np.array([x_cur_true, y_cur_true])))
        # convert to unit length
        goal_vector = goal_vector / np.linalg.norm(goal_vector)
        print("goal_vector " , goal_vector, "Goal: ", goal_xy, "dist", np.linalg.norm(goal_vector))
        
        # get the frame from unity
        frame = get_next_frame(unityframe_shm).to(device)
        # plt.close()
        # plt.imshow(frame.cpu().squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
        # plt.show()
        with torch.no_grad():
            m = vae.encode(frame.unsqueeze(0))
            
            # p = vae.latent2position_mapping(m)
            # ax.scatter(p[0][0].item()*55, -p[0][1].item()*55, c="r", s=20, alpha=.8)
            # ax.scatter(p[0][0].item()*55, p[0][1].item()*55, c="r", s=20, alpha=.2)
            
            
            # m_pred = vae.position2latent_mapping.weight.data@(torch.tensor([x_cur_true, y_cur_true, 90]).to(device)/55.0)
            # p_pred = vae.latent2position_mapping.weight.data@(m_pred)
            # ax.scatter(p_pred[0].item()*55, p_pred[1].item()*55, c="y", s=20, alpha=.2)
            
            m = m.squeeze(0).cpu().numpy()

            # z_batch = vae.encode(frame.unsqueeze(0))
            # pos_batch = torch.Tensor([x_cur_true, y_cur_true, 90]).unsqueeze(0).to(device) / 55.0
            # x_reconstr_batch = vae.decode(z_batch)
            # z_batch_pred = vae.position2latent_mapping(pos_batch)
            # print(vae.position2latent_mapping.bias.data)
            # print(z_batch_pred[0])
            # print(vae.position2latent_mapping.weight.data@pos_batch[0] +vae.position2latent_mapping.bias.data)
            # x_reconstr_batch_pred = vae.decode(z_batch_pred)
            # fig, ax = plt.subplots(1,3)
            # ax[0].imshow(frame.detach().cpu().numpy().squeeze(), cmap='gray', vmin=0, vmax=1)
            # ax[0].set_title("$I$")
            # ax[0].axis('off')
            # ax[1].imshow(x_reconstr_batch[0].detach().cpu().numpy().squeeze(), cmap='gray', vmin=0, vmax=1)
            # ax[1].set_title(r"$f_{decoder}(f_{encoder}(I_{i})) = \hat{I_{i}}$")
            # ax[1].axis('off')
            # ax[2].imshow(x_reconstr_batch_pred[0].detach().cpu().numpy().squeeze(), cmap='gray', vmin=0, vmax=1)
            # ax[2].set_title(r"$f_{decoder}(p^{T}H) = \hat{I_{i}}$")
            # ax[2].axis('off')
            # plt.savefig(f"./figs/pos2z_reconstruction_{i}.svg")
            # plt.show()            
        
        
        # if i > 0:
        #     print(np.linalg.norm(np.array([x_cur_true, y_cur_true]) - np.array([x_prv_true, y_prv_true])))
        
        # predict step
        vel = np.array([goal_vector[0], goal_vector[1], 0])
        # tau = .015
        tau = 5
        s_cur_given_m_prv_mean = tau*vel + s_prv_given_m_prv_mean
        s_cur_given_m_prv_cov = Q + s_prv_given_m_prv_cov
        print("Predicte mean: ", s_cur_given_m_prv_mean)
        
        # Update Step
        # Kalman Gain
        
        # R = np.eye(24)*(nll*2)
        nll = nlog_likelihood(H@(s_cur_given_m_prv_mean/55.), mu_mle, sigma_mle)
        R = np.eye(24)*3
        # R = np.ones_like(R)
        # biases = vae.position2latent_mapping.bias.data.cpu().numpy()
        K = s_cur_given_m_prv_cov@H.T @ np.linalg.inv(H @ s_cur_given_m_prv_cov @ H.T + R)
        # Updated state estimate
        
        # s_cur_given_m_prv_mean = np.array([x_cur_true, y_cur_true, 0])
        
        # s_cur_given_m_prv_mean *= np.array([1,-1,1])
        innovation = m-H@(s_cur_given_m_prv_mean/55.)
        # s_cur_given_m_prv_mean *= np.array([1,1,1])
        # innovation = m-m
        measured_pos = K @ innovation
        measured_pos *= np.array([1,-1/55,1])
        mu_updated = s_cur_given_m_prv_mean/55. + measured_pos
        print("measured_pos: " , measured_pos*55.)
        # mu_updated *= np.array([1,-1,1])
        # mu_updated = s_cur_given_m_prv_mean + K @ (m - m)
        s_cur_given_m_cur_mean = mu_updated*55.
        # s_cur_given_m_cur_mean = mu_updated
        # Updated covariance estimate
        Sigma_updated = (np.eye(3) - K @ H) @ s_cur_given_m_prv_cov
        s_cur_given_m_cur_cov = Sigma_updated
        # print("Updated mean: ", s_cur_given_m_cur_mean, )#"diff: ", m-H@s_cur_given_m_prv_mean)
        # print("m: ", m, "NLL: ", nll, "avg diff: ", np.mean(innovation))
        
        s_prv_given_m_prv_mean = s_cur_given_m_cur_mean
        s_prv_given_m_prv_cov = s_cur_given_m_cur_cov
        KLd = KL_divergence_general(s_cur_given_m_prv_mean, s_cur_given_m_prv_cov,
                                    s_cur_given_m_cur_mean, s_cur_given_m_cur_cov,
                                     )

        
        val1 = nll
        val2 = np.linalg.norm(innovation)
        val3 = KLd
        if i > 0:
            ax2[0].plot((i-1, i), (val1_prv, val1), color='k')
            ax2[1].plot((i-1, i), (val2_prv, val2), color='k')
            ax2[2].plot((i-1, i), (val3_prv, val3), color='k')
        val1_prv = val1
        val2_prv = val2
        val3_prv = val3
        val1_all.append(val1)
        val2_all.append(val2)
        val3_all.append(val3)
        # print(min(np.array(val1_all)), max(np.array(val1_all)))
        # print(min(np.array(val2_all)), max(np.array(val2_all)))
        # print(min(np.array(val3_all)), max(np.array(val3_all)))
        
        
        # draw real trajectory
        ax.scatter(x_cur_true, y_cur_true, c="k", s=4, alpha=.8)
        # if i != 0 and i == 1:
        ax.plot([x_prv_true, x_cur_true], [y_prv_true, y_cur_true], c="k", alpha=.3)

        draw_uncertainty(ax, s_cur_given_m_prv_mean, s_cur_given_m_prv_cov, fc='purple', label=i)
        draw_uncertainty(ax, s_cur_given_m_cur_mean, s_cur_given_m_cur_cov, fc='green', label="")
        ax.arrow(s_cur_given_m_prv_mean[0], s_cur_given_m_prv_mean[1], 
                    s_cur_given_m_cur_mean[0]-s_cur_given_m_prv_mean[0], 
                    s_cur_given_m_cur_mean[1]-s_cur_given_m_prv_mean[1],  
                    fc='k', ec='k', alpha=.3, length_includes_head=True, 
                    head_width=2, head_length=2, )
        # plt.arrow(x_prv_true, y_prv_true, x_cur_true-x_prv_true, y_cur_true-y_prv_true, head_width=2, head_length=2, fc='k', ec='k', alpha=.3)
        
        
        angular_vel = 0
        # sample velocity noise from Q, scale to velocity (cov is for posoition noice)
        motor_noise_x = np.random.normal(0, Q[0, 0])*1
        motor_noise_y = np.random.normal(0, Q[1, 1])*1
        x_vel = goal_vector[0]*vel_step_size + motor_noise_x
        y_vel = goal_vector[1]*vel_step_size + motor_noise_y
        # print("X: ", x_vel,  goal_vector[0]*vel_step_size, motor_noise_x)
        # print("Y: ", y_vel,  goal_vector[1]*vel_step_size, motor_noise_y)
        requests.post(f"{base_url}/unityinput/Move,{x_vel:.2f},{angular_vel},{-y_vel:.2f}")
        # requests.post(f"{base_url}/unityinput/Teleport%2C{-10:.3f}%2C{-10:.3f}%2C{270:.3f}")

        x_prv_true, y_prv_true = x_cur_true, y_cur_true
        # s_prv_given_m_prv_mean = np.array([x_cur_true, y_cur_true, 0])
        i += 1
        time.sleep(.4)
        unityout_shm.reset_reader()
        print()
        print()
            
def get_vae_model():
    latent_dims = 24
    vae = VAE(latent_dims)
    vae.train()
    vae.load_state_dict(torch.load('./vaelulGAMMA_E3.pth'))
    return vae

# To run the async function
if __name__ == "__main__":
    # initate()
    # start_unity()
    # start_session()
    
    # term_delete()
    vae = get_vae_model()
    z = np.load("./z_embeddings.npy")
    mu_mle, sigma_mle = mle(z)
    device = get_device()
    
    kalman_filter(vae, mu_mle, sigma_mle, device)
    pass