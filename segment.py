"""
Segmentation with the Representation Learning model

RL network identifies motion features.
We now use these features (gained from two image inputs)
to re-train a small part of the network for the supervised segmentation.

Test three different models:
1. Retrain second half
2. Train full network on segmentation only
3. Only train last layer (output)

"""

import json
import argparse
# import matplotlib.pyplot as plt

import torch

import data_utils
from models.ResUNet import ResUNet
from models.methods import PyTorchModel

if torch.cuda.is_available():
    DEVICE = 'gpu'
else:
    DEVICE = 'cpu'


def freeze_layers(model, layer):
    """
    Freeze all the layers before 'layer' by setting the param.requires_grad to
    False.
    Returns the frozen model.
    """
    for num, param in enumerate(model.children.parameters()):
        if num < layer:
            param.requires_grad = False
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",
                        help="Name of the config file inside ./config/")

    args = parser.parse_args()

    # config variables
    with open("configs/" + args.config + ".json", 'r') as config_json:
        config = json.load(config_json)

    model_name = config["model_name"]
    weights_path = config["weights_path"]
    test_path = config["test_path"]
    results_path = config["results_path"]
    image_size = config["image_size"]

    print(json.dumps(config, indent=4), flush=True)

    test_data = data_utils.load_data(test_path, (image_size, image_size))

    model = ResUNet(in_channels=2,
                    out_channels=1,
                    block_sizes=[32, 64, 128, 256, 512, 1024],
                    depths=[2, 3, 5, 3, 2])

    model.load_state_dict(torch.load(weights_path))
    model.to(DEVICE)

    model = PyTorchModel(model, 0)
    logs = model.test(test_data)

    model.save_model(model_name)
    model.save_logs(results_path)
