import os
import h5py
import torch
import cv2
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

class UnityLocalizationDataset(Dataset):
    def __init__(self, path, proportion=0.8, train=True):
        # Open the HDF5 files
        frames_fullfname = os.path.join(path, 'unitycam.hdf5')
        locations_fullfname = os.path.join(path, 'unity_output.hdf5')
        
        self.frames_file = h5py.File(frames_fullfname, 'r')['frames']
        self._frame_keys = tuple(self.frames_file.keys())
        
        self.frame_indices, self.frame_locations = self.load_data(frames_fullfname, 
                                                                  locations_fullfname, 
                                                                  proportion, train)
        
    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        # Retrieve the frameidx from the frame_packages
        frame_idx = self.frame_indices.iloc[idx].name
        location = torch.tensor(self.frame_locations.loc[frame_idx], dtype=torch.float32)
        location /= 55.0 #normalize to range [-1, 1]
        
        frame_key = f"frame_{frame_idx:06}"
        frame = self.frames_file[frame_key][()]
        frame = cv2.imdecode(np.frombuffer(frame.tobytes(), np.uint8), 
                             cv2.IMREAD_COLOR) 

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize the frame to a fixed size (e.g., 224x224)
        frame = cv2.resize(frame, (224, 224))
        # Convert frame to float32, normalize to range [0, 1]
        frame = torch.tensor(frame, dtype=torch.float32) / 255.0
        frame = frame.unsqueeze(0)
        
        # plt.imshow(frame, cmap='gray', vmin=0, vmax=1)
        # plt.title(f"Location: {location.numpy()}")
        # plt.show()
        
        return frame, location

    def load_data(self, frames_fullfname, locations_fullfname, proportion, train):
        frame_indices = pd.read_hdf(frames_fullfname, key='frame_packages')
        frame_indices.set_index("ID", drop=True, inplace=True)
        
        frame_locations = pd.read_hdf(locations_fullfname, key='unityframes')
        frame_locations.set_index("ID", drop=True, inplace=True)
        frame_locations = frame_locations.loc[:,['X', 'Z', 'A']]
        
        index = int(len(frame_indices)*proportion)
        print(len(frame_indices))
        # print(index)
        # start from the beginning for train, from the end for test
        if train:
            frame_indices = frame_indices.iloc[:index]
        else:
            frame_indices = frame_indices.iloc[-index:]
        print("----")
        print(frame_indices)
        print(frame_locations)
        print("----")
        return frame_indices, frame_locations

    def close(self):
        self.frames_file.close()

if __name__ == "__main__":
    # Define the path to the dataset
    path = '/Users/loaloa/homedataAir/phd/ratvr/VirtualReality/data/2024-06-30_15-39_mlDatasetBig'
    
    # Create the dataset object
    D_train = UnityLocalizationDataset(path, proportion=.8, train=True)
    # Get the length of the dataset
    print(len(D_train))
    # Get the first item in the dataset
    frame, output = D_train[0]
    print(frame.shape, output.shape)
    
    #same for the test set
    D_test = UnityLocalizationDataset(path, proportion=.2, train=False)
    print(len(D_test))
    frame, output = D_test[0]
    print(frame.shape, output.shape)

    # # Create a DataLoader
    # dataloader = DataLoader(D_train, batch_size=4, shuffle=True)
    # # Example: Iterate over the DataLoader
    # for i, (frames, outputs) in enumerate(dataloader):
    #     print(f"Batch {i}:")
    #     print(f"Frames shape: {frames.shape}")
    #     print(f"Outputs shape: {outputs.shape}")
    #     # Here you would typically forward the batch through your model
    # # Remember to close the dataset files when done
    # D_train.close()
