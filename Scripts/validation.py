import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from einops import rearrange, repeat
from tqdm import tqdm
import numpy as np

from module import FullNetwork, SDFDecoder, VoxelCNNEncoder
import config
from dataset.utils.torch_load import create_test_validation_data_loader, VoxelSDFDataset

import logging

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure
from skimage.draw import ellipsoid

def create_marching_cube_mesh(voxel_grid : np.ndarray, network : FullNetwork, config : config.Config):
    x = torch.linspace(0, 1, 100)
    y = torch.linspace(0, 1, 100)
    z = torch.linspace(0, 1, 100)
    x, y, z = torch.meshgrid(x, y, z, indexing='ij')
    points = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)
    point_batch = torch.split(points, 200000)

    latent_code = network.encoder(torch.from_numpy(voxel_grid))
    sdfs = torch.zeros(0)
    for points in point_batch:
        points = points.to(device=config.device)
        latent_code_repeated = repeat(latent_code, "l -> s l", s=points.shape[0])
        sdfs = torch.cat((sdfs, network.decoder(latent_code_repeated, points)))
    sdfs.flatten()
    
    sdfs = sdfs.cpu().numpy().reshape(100, 100, 100)
    verts, faces, normals, values = measure.marching_cubes(sdfs, level=0)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)
    
    ax.set_xlim(0, sdfs.shape[0])
    ax.set_ylim(0, sdfs.shape[1])
    ax.set_zlim(0, sdfs.shape[2])
    
    plt.show()


def validation(network : FullNetwork, validation_loader : DataLoader, config : config.Config):
    network.eval()
    
    criterion = nn.MSELoss()
    losses = torch.zeros(0)
    
    with torch.no_grad():    
        for i, (voxel_tensor, point, sdf) in tqdm(enumerate(validation_loader, desc="Validation")):
            point = rearrange(point, "b s c -> (b s) c")
            sdf = rearrange(sdf.unsqueeze(2), "b s c -> (b s) c")

            voxel_tensor = voxel_tensor.to(device=config.device)
            point = point.to(device=config.device)
            sdf = sdf.to(device=config.device)
            
            voxel_tensor = voxel_tensor.unsqueeze(1)

            latent = network.encoder(voxel_tensor)                
            latent = repeat(latent, "b l -> b s l", s=config.num_points_per_minor_batch)
            latent = rearrange(latent, "b s l -> (b s) l")
            sdf_pred = network.decoder(latent, point)

            loss = criterion(sdf_pred, sdf)
            losses = torch.cat((losses, loss))

    losses = losses.numpy()
    loss_avg = np.average(losses)
    loss_stddev = np.std(losses)
    print(f"Average of losses: {loss_avg}")
    print(f"Standard deviation of losses: {loss_stddev}")

if __name__ == "__main__":
    cfg = config.Config()
    network = FullNetwork(config=cfg)
    _, validation_dataloader = train_dataloader, validation_loader = create_test_validation_data_loader(
        dataset_dir=cfg.dataset_path, 
        dataset_config_file="dataset/config.json",
        num_sdf_samples_per_item=cfg.num_points_per_minor_batch
    )
    validation(network, validation_dataloader, cfg)
