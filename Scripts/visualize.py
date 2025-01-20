import numpy as np
from skimage import measure
import trimesh
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import module
import config

from tqdm import tqdm

def get_sdf_from_model(full_network : module.FullNetwork, points : np.ndarray, sdfs : np.ndarray, voxel_grid : np.ndarray, config : config.Config):
    encoder = full_network.encoder
    decoder = full_network.decoder
    
    voxel_grid = torch.from_numpy(voxel_grid).unsqueeze(0).unsqueeze(1) # Shape: 1x1x64x64x64
    voxel_grid = voxel_grid.to(config.device)

    # Initialize to mesh grid points
    x, y, z = np.mgrid[-2 : 2 : 0.04, -2 : 2 : 0.04, -2 : 2 : 0.04]
    x = torch.tensor(x)
    y = torch.tensor(y)
    z = torch.tensor(z)

    # Change the mesh grid points from 100x100x100x3 to 1000000x3
    points = torch.stack((x, y, z), dim=3).view(-1, 3)
    points = points.to(config.device)
    # Split the points into 4 seperate splits to prevent running out of memory
    points_splits = points.split(250000)

    # Encode the voxel grid
    latent_code = encoder(voxel_grid) # Return tensor with shape 1x256
    latent_code = repeat(latent_code, "b n -> b s n", s=250000) # Convert the latent code from 1x256 to 1x250000x256
    latent_code = rearrange(latent_code, "b n -> (b s) n") # Convert the latent code from 1x250000x256 to 250000x256

    sdfs = torch.zeros(0, device="cpu")
    for points_split in points_splits:
        sdf = torch.cat(sdf, decoder(latent_code, points_split).cpu())

    sdfs = sdfs.view((100, 100, 100, 3))

def generate_sdf_objs(dataset_json_path, fullmodel : module.FullNetwork, config : config.Config, categories="All"):
    dataset_json = dict()
    with open(dataset_json_path, "r") as f:
        dataset_json = json.load(f)
    if categories == "All":
        categories = list(dataset_json.keys())
    validation_models_np_file = []

    for category in categories:
        split_indices = dataset_json[category]["split"][1]
        for i in split_indices:
            validation_models_np_file.append(dataset_json[category]["models"][i])

    with torch.no_grad():
        for validation_model_np_file in tqdm(validation_models_np_file, desc="model"):
            # Get model data from npz file
            fulldata = np.load(validation_model_np_file)
            points, sdfs, voxel_grid = fulldata["points"], fulldata["sdfs"], fulldata["voxel_grid"]

            # Get the sdf values from model
            sdf = get_sdf_from_model(fullmodel, points, sdfs, voxel_grid)
            
            # Marching Cube
            verts, faces, normals, _ = measure.marching_cubes(sdf, 0)
            
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, face_normals=normals)
            with open(os.path.splitext(validation_model_np_file)[0] + ".obj", "w") as f:
                mesh.export(f, "obj")

if __name__ == "__main__":
    x, y, z = np.mgrid[-2 : 2 : 0.04, -2 : 2 : 0.04, -2 : 2 : 0.04]
    x = torch.tensor(x)
    y = torch.tensor(y)
    z = torch.tensor(z)
    points = torch.stack((x, y, z), dim=3).view(-1, 3)
    print(points.view(100, 100, 100, 3))
