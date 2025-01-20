import torch

import os
import json
import logging

class Config:
    def __init__(self, config_file="Scripts/config.json"):
        with open(config_file) as cfile:
            self.config = json.load(cfile)
        
        self.input_voxel_grid_size = self.config["input_voxel_grid_size"]
        self.decoder_hidden_dimension = self.config["decoder_hidden_dimension"]
        self.decoder_num_hidden_layers = self.config["decoder_num_hidden_layers"]
        self.latent_dimension = self.config["latent_dimension"]
        self.num_sdf_samples = self.config["num_sdf_samples"]
        self.num_points_per_minor_batch = self.config["num_points_per_minor_batch"]
        self.dataset_path = self.config["dataset_path"]
        self.shapenet_categories = self.config["shapenet_categories"]
        self.seperate_ratio = self.config["seperate_ratio"]
        self.train_epoch = self.config["train_epoch"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.learning_rate = self.config["learning_rate"]
        self.check_point_filename = self.config["check_point_filename"]
        self.loss_filename = self.config["loss_filename"]
        self.is_train = self.config["is_train"]
        self.is_validation = self.config["is_validation"]

    def log(self): 
        logging.info(self.config)

if __name__ == "__main__":
    config = Config()
    logging.basicConfig(level=logging.DEBUG)
    for category in config.shapenet_categories:
        category_data_pickle_name = os.path.splitext(config.shapenet_pickle_name)[0] + "_" + category + ".pkl"
        logging.debug(category_data_pickle_name)