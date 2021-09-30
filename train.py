"""
Main python script for the training of the dynamic texture model
"""
###############################################################################

# Written by Ryan Smith
# ryan.smith@ucdconnect.ie
# github.com/ryan597/DynamicTextureWaves

###############################################################################

# Python imports
import argparse
import json
# import seaborn as sns
from os.path import exists

# Pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim

# My imports
import utils
from models.ResUNet import ResUNet

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

###############################################################################
# Parse arguments
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",
                        help="Name of the config file inside ./config/")
    args = parser.parse_args()

    IMAGE_SIZE = 300

    # config variables
    with open("configs/" + args.config + ".json", 'r') as config_json:
        config = json.load(config_json)
    model_name = config["model_name"]
    weights_path = config["weights_path"]
    train_path = config["train_path"]
    results_path = config["results_path"]
    imaug = config["imaug"]

    # Loading datasets
    train_data = utils.load_data(train_path, IMAGE_SIZE)

    # Loading model
    model = ResUNet(in_channels=2,
                    out_channels=1,
                    layer_size=[64, 128, 256, 512, 1024])

    model.to(DEVICE)

    if exists(weights_path):
        model.load_state_dict(torch.load(weights_path))

    # utils.show_samples(train_data)

    # Training model

    # Results

###############################################################################
# Save Models

# #torch.save(model.state_dict(), f"DT_model_spill_128.pth")
# torch.save(model.state_dict(), f"DT_model_spill_400_pretrained.pth")

# #torch.save(model.state_dict(), f"DT_model_plunge_128.pth")
# torch.save(model.state_dict(), f"DT_model_plunge_400_pretrained.pth")

# torch.save(model.state_dict(), f"DT_model_nonbreaking_128.pth")
# torch.save(model.state_dict(), f"DT_model_nonbreaking_400.pth")

###############################################################################


class PyTorchModel():
    """
    Class for PyTorch models to easily train, validate and test the model.

    Arguments:
        model (nn.Module): An instance of nn.Module which has a defined
            forward() method
        epochs (int): The number of epochs to train for. Default value 10.
        learning_rate (double): The multiplication parameter to be passed when
            updating weights in backpropagation. Controls how large the weights
            updates can be. Default value 0.01
        criterion (torch.autograd.Function): Should be an instance of one of
            the pytorch loss functions or a callable function which returns a
            loss value. A custom function should use torch operations
            (in loss calculation and in the forward pass) for autograd.
            Otherwise a custom class which inherits from
            torch.autograd.Function and defines both forward and backward
            methods. Default value torch.nn.L1Loss()
        optimizer (torch.optim.optimizer.Optimizer):
        scheduler (): Function to call when changing the learning rate.
            Default value None

    Methods:
        train_model:
        save_model:
        evaluate_model:
        predict:
        update_logs:
        get_logs:

    """
    def __init__(self,
                 model,
                 epochs=10,
                 learning_rate=0.01,
                 criterion=nn.L1Loss,
                 optimizer=optim.Adam,
                 scheduler=None):
        self.logs = {"loss": [], "val_loss": [], "epoch": [], "lr": []}
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_model(self):
        pass

    def save_model(self):
        pass

    def evaluate_model(self):
        pass

    def predict(self, x):
        pass

    def update_logs(self):
        pass

    def get_logs(self):
        pass
