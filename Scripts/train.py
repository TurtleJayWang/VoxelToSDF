import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from einops import rearrange, repeat

from module import FullNetwork, VoxelCNNEncoder, SDFDecoder
import config

import sys
sys.path.append("./")
from dataset.utils.torch_load import VoxelSDFDataset, create_test_validation_data_loader

import os
import numpy as np
from tqdm import tqdm
import logging
import pickle
import glob
import argparse
import matplotlib.pyplot as plt

def loss(pred, target, clamp_dist=0.1) -> torch.Tensor:
    clamped_pred = torch.clamp(pred, -torch.ones(pred.shape) * clamp_dist, torch.ones(pred.shape) * clamp_dist)
    clamped_target = torch.clamp(target, -torch.ones(pred.shape) * clamp_dist, torch.ones(pred.shape) * clamp_dist)
    return torch.abs(clamped_pred - clamped_target)

class ModelTrainer:
    def __init__(self, train_dataloader : DataLoader, config : config.Config):
        self.epoch = config.train_epoch
        self.epoch_start = 0
        self.dataloader = train_dataloader
        self.network = FullNetwork(config=config)
        self.device = config.device
        self.checkpoint_filename = config.check_point_filename
        self.losses : torch.Tensor = torch.zeros(0)

        self.network = self.network.to(device=self.device)

        self.config = config

        self.criterion = loss
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)

    def train(self):
        self.load_loss()

        self.network.train()
        for k in tqdm(range(self.epoch), desc="Epoch", position=2):
            batch_loss = torch.zeros(1).to(self.config.device)
            length = 0
            for i, (voxel_tensor, point, sdf) in tqdm(enumerate(self.dataloader), desc="Batch", position=3, ncols=80, leave=False):
                point = rearrange(point, "b s c -> (b s) c")
                sdf = rearrange(sdf, "b s c -> (b s) c")

                voxel_tensor = voxel_tensor.to(device=self.device)
                point = point.to(device=self.device)
                sdf = sdf.to(device=self.device)
                
                voxel_tensor = voxel_tensor.unsqueeze(1)

                latent = self.network.encoder(voxel_tensor)                
                latent = repeat(latent, "b l -> b s l", s=self.config.num_points_per_minor_batch)
                latent = rearrange(latent, "b s l -> (b s) l")
                sdf_pred = self.network.decoder(latent, point)

                loss = self.criterion(sdf_pred, sdf)
                batch_loss += loss
                length += 1

                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()
            
            self.losses = torch.cat((self.losses, (batch_loss.cpu() / length)))

            self.save_parameters(self.epoch_start + k)
            self.save_loss(self.epoch_start + k)
        
    def save_parameters(self, k):
        name = f"{os.path.splitext(self.checkpoint_filename)[0]}.{k}{os.path.splitext(self.checkpoint_filename)[1]}"
        with open(name, "wb") as cp_f:
            print("Checkpoint saved to", name)
            torch.save(self.network.state_dict(), cp_f)
        
    def load_parameters(self):
        names = glob.glob(f"{os.path.splitext(self.checkpoint_filename)[0]}.*{os.path.splitext(self.checkpoint_filename)[1]}")
        if len(names):
            max_epoch = 0
            load_checkpoint_file_name = ""
            for name in names:
                epoch = int(name.split(".")[-2])
                if epoch > max_epoch:
                    max_epoch = epoch
                    load_checkpoint_file_name = name
            self.epoch_start = max_epoch + 1
            with open(load_checkpoint_file_name, "rb") as f:
                self.network.load_state_dict(torch.load(f))
                print("Checkpoint loaded", load_checkpoint_file_name)
                self.network.to(self.config.device)
    
    def save_loss(self, k):
        name = f"{os.path.splitext(self.config.loss_filename)[0]}.{k}{os.path.splitext(self.config.loss_filename)[1]}"
        with open(name, "wb") as f:
            print("Losses saved to", name)
            pickle.dump(self.losses, f)

    def load_loss(self):
        names = glob.glob(f"{os.path.splitext(self.config.loss_filename)[0]}.*{os.path.splitext(self.config.loss_filename)[1]}")
        if len(names):
            max_epoch = 0
            load_loss_file_name = ""
            for name in names:
                epoch = int(name.split(".")[-2])
                if epoch > max_epoch:
                    max_epoch = epoch
                    load_loss_file_name = name
            self.epoch_start = max_epoch + 1
            with open(load_loss_file_name, "rb") as f:
                self.losses = pickle.load(f)
                print("Losses loaded", load_loss_file_name)
        else: self.losses = torch.zeros(0)

    def visualize_loss(self):
        plt.plot(np.arange(len(self.losses.cpu().detach().numpy())), self.losses.cpu().detach().numpy())
        plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    cfg = config.Config()
    train_dataloader, _ = create_test_validation_data_loader(
        dataset_dir=cfg.dataset_path,
        batch_size=32,
        dataset_config_file="dataset/config.json",
        num_sdf_samples_per_minor_batch=cfg.num_points_per_minor_batch
    )
    
    model_trainer = ModelTrainer(train_dataloader=train_dataloader, config=cfg)

    #model_trainer.load_loss()
    #model_trainer.visualize_loss()

    model_trainer.load_parameters()
    model_trainer.train()
    model_trainer.visualize_loss()

    logging.basicConfig(level=logging.DEBUG)
