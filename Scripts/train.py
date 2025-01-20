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
import argparse

class ModelTrainer:
    def __init__(self, train_dataloader : DataLoader, config : config.Config):
        self.epoch = config.train_epoch
        self.dataloader = train_dataloader
        self.network = FullNetwork(config=config)
        self.device = config.device
        self.checkpoint_filename = config.check_point_filename
        self.losses = []

        self.network = self.network.to(device=self.device)

        self.config = config

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)

    def train(self):
        self.load_loss()

        self.network.train()
        for k in tqdm(range(self.epoch), desc="Epoch", position=2):
            batch_loss = 0
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
                batch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()
                length = i + 1
            
            self.losses.append(batch_loss / length)

            if k % 10 == 0 and k != 0:
                self.save_parameters()
                self.save_loss()
        
    def save_parameters(self):
        with open(self.checkpoint_filename, "wb") as cp_f:
            torch.save(self.network.state_dict(), cp_f)
        
    def load_parameters(self):
        if os.path.exists(self.checkpoint_filename):
            with open(self.checkpoint_filename, "b+r") as cp_f:
                self.network.load_state_dict(torch.load(cp_f))
    
    def save_loss(self):
        with open(self.config.loss_filename, "wb") as f:
            pickle.dump(self.losses, f)

    def load_loss(self):
        if os.path.exists(self.config.loss_filename):
            with open(self.config.loss_filename, "rb") as f:
                self.losses = pickle.load(f)
        else: self.losses = []

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    cfg = config.Config()
    train_dataloader, validation_loader = create_test_validation_data_loader(
        dataset_dir=cfg.dataset_path,
        batch_size=32,
        dataset_config_file="dataset/config.json",
        num_sdf_samples_per_minor_batch=cfg.num_points_per_minor_batch
    )
    
    model_trainer = ModelTrainer(train_dataloader=train_dataloader, config=cfg)

    model_trainer.load_parameters()
    model_trainer.train()
    model_trainer.save_parameters()

    logging.basicConfig(level=logging.DEBUG)
