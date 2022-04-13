"""
Main python script for the training of the dynamic texture model
"""
###############################################################################

# Written by Ryan Smith
# ryan.smith@ucdconnect.ie
# github.com/ryan597/RepresentationLearningWaves

###############################################################################

# Python imports
import os
import argparse
import json
import numpy as np
# import seaborn as sns
from os.path import exists

# Pytorch imports
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Other imports
import data_utils
from models.ResUNet import ResUNet
from models.methods import PyTorchModel

# Randomness must be disabled for distributed training!
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def init_process(rank, size, backend="nccl"):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)


def cleanup():
    dist.destroy_process_group()


###############################################################################
# Parse arguments
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",
                        help="Name of the config file inside ./config/")
    parser.add_argument("-r", "--rank", type=int,
                        help="Local rank or device number")
    args = parser.parse_args()

    # config variables
    with open("configs/" + args.config + ".json", 'r') as config_json:
        config = json.load(config_json)
    
    if (args.rank == 0):
        print(json.dumps(config, indent=4), flush=True)

    model_name = config["model_name"]
    weights_path = config["weights_path"]
    train_path = config["train_path"]
    results_path = config["results_path"]
    valid_path = config['valid_path']
    image_width = config["image_width"]
    image_height = config["image_height"]
    imaug = config["imaug"]

    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    criterion = getattr(torch.nn, config["criterion"])
    optimizer = getattr(torch.optim, config["optimizer"])
    scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"])

    # Must run the script with this many different ranks
    # If world_size=2, then you must run in the terminal
    #       python main.py -c config -r 0 &
    #       python main.py -c config -r 1
    # The process initialiser halts the program until all processes have joined
    world_size = torch.cuda.device_count()
    rank = args.rank
    init_process(rank, world_size)

    # Loading datasets
    train_data = data_utils.load_data(train_path, rank, world_size, (image_height, image_width), batch_size=5)
    valid_data = data_utils.load_data(valid_path, rank, world_size, (image_height, image_width), batch_size=5)
    #data_utils.show_samples(train_data)

    # Loading model
    model = ResUNet(in_channels=2,
                    out_channels=1,
                    block_sizes=[32, 64, 128, 256, 512, 1024],
                    depths=[2, 3, 5, 3, 2])

    if exists(weights_path):
        model.load_state_dict(torch.load(weights_path))

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  ## Sync BatchNorm for MultiGPU
    model = model.to(f"cuda:{rank}")
    DPPmodel = DDP(model, device_ids=[rank])
    model = PyTorchModel(DPPmodel,
                         rank=rank,
                         epochs=epochs,
                         learning_rate=learning_rate,
                         criterion=criterion,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         mode="min",
                         factor=0.1,
                         patience=2,
                         threshold=0.001,
                         verbose=True)  # scheduler kwarg

    # Training model
    with DPPmodel.join():
        logs = model.train_model(train_data, valid=valid_data)

    if rank == 0:
        model.save_model(model_name)
        model.save_logs(results_path)

    cleanup()
