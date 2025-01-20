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

class VoxelCNNEncoderForRender(nn.Module):
    def __init__(self, input_size=(128, 128, 128), latent_size=256):
        super().__init__()
        
        self.input_size = input_size

        # Add batch normalization for better training stability
        self.encoder = nn.Sequential(
            # First block
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout3d(0.3),
            
            # Second block
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout3d(0.3),
            
            # Third block
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout3d(0.3),
            
            nn.Flatten()
        )
        
        # Calculate the size of the flattened features
        # Input: 128x128x128
        # After first block: 128x128x128 -> 64x64x64
        # After second block: 16x16x16 -> 8x8x8
        # After third block: 2x2x2 -> 1x1x1
        flattened_size = 64 * 1 * 1 * 1
        
        # Add final linear layers with skip connection
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, latent_size)
        
    def forward(self, x):
        # Main encoder path
        features = self.encoder(x)
        
        # Final fully connected layers with skip connection
        x = self.fc1(features)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.fc2(x)
        
        return x


def create_mesh_from_sdf(sdf):
    x, y ,z = np.mgrid[-1:1:0.02, -1:1:0.02, -1:1:0.02]
    x = torch.tensor(x)
    y = torch.tensor(y)
    z = torch.tensor(z)
    points = torch.stack((x, y, z), dim=-1).reshape(-1, 3)
    sdfs = sdf(points)
    print(sdfs.shape)
    verts, faces, normals, _ = measure.marching_cubes(sdfs, 0)
    return verts, faces, normals

def save_to_obj(sdf, filename):
    verts, faces, normals = create_mesh_from_sdf(sdf)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, face_normals=normals)
    with open(filename, "w") as f:
        mesh.export(f, "obj")

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
    
    encoder = VoxelCNNEncoderForRender().to(config.device)
    encoder.encoder = fullmodel.encoder.encoder
    encoder.fc1 = fullmodel.encoder.fc1
    encoder.fc2 = fullmodel.encoder.fc2

    with torch.no_grad():
        for validation_model_np_file in tqdm(validation_models_np_file, desc="model"):
            validation_model_np_data = np.load(os.path.join("data", validation_model_np_file))
            voxel_grid = validation_model_np_data["voxel_grid"]
            voxel_tensor = torch.from_numpy(voxel_grid).to(config.device).float().unsqueeze(0).unsqueeze(1).to(config.device)
            
            latent_vector = encoder(voxel_tensor)
            latent_vector = repeat(latent_vector, "b l -> b n l", n=250000)
            latent_vector = rearrange(latent_vector, "b n l -> (b n) l")
            
            x, y ,z = np.mgrid[-1:1:0.02, -1:1:0.02, -1:1:0.02]
            x = torch.tensor(x)
            y = torch.tensor(y)
            z = torch.tensor(z)
            points = torch.stack((x, y, z), dim=-1).reshape(-1, 3).float()
            points_lists = points.split(250000)

            sdfs = torch.tensor([]).to(device=config.device)
            for points_tensor in points_lists:
                new_sdf = fullmodel.decoder(latent_vector, points_tensor.to(config.device))
                sdfs = torch.cat((sdfs, new_sdf))

            result_sdf = np.zeros((100, 100, 100))
            for i in range(points.shape[0]):
                a = int(points[i][0] + 1) * 50
                b = int(points[i][1] + 1) * 50
                c = int(points[i][2] + 1) * 50
                result_sdf[a][b][c] = sdfs[i]

            verts, faces, normals, _ = measure.marching_cubes(result_sdf, 0)
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, face_normals=normals)
            with open(os.path.splitext(validation_model_np_file)[0] + ".obj", "w") as f:
                mesh.export(f, "obj")

if __name__ == "__main__":  
    cfg = config.Config()
    fullnetwork = module.FullNetwork(cfg).to(cfg.device)
    with open("checkpoints/checkpoint.pkl", "rb") as f:
        fullnetwork.load_state_dict(torch.load(f))
    generate_sdf_objs(
        dataset_json_path="data/dataset.json",
        fullmodel=fullnetwork,
        config=cfg
    )
