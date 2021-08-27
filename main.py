###############################################################################

# Written by Ryan Smith
# ryan.smith@ucdconnect.ie

###############################################################################

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression as lr
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.tree import export_text
# from sklearn.ensemble import RandomForestClassifier as rfc

import utils

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

    # config variables
    # model_name
    # weights
    # train_path
    # features_path
    # labels_path
    # results


###############################################################################
# Data Pipeline
class InputImages(Dataset):
    def __init__(self, path, transform):
        self.file_path = path
        self.files = glob.glob(self.file_path+'/*.png')
        self.transform = transform
        self.dataset_len = len(files)

    def __getitem__(self, index):

        return input_imgs, third_img

    def __len__(self):
        return self.dataset_len


def load_data(path, image_size, batch_size=1, shuffle=True):
    transform = get_transform(image_size=image_size)
    dataset = InputImages(path, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=8)
    return dataloader

###############################################################################
# Loading Models

###############################################################################
# Training

###############################################################################

plunge_model = UNet(2, 1, [64, 128, 256, 512, 1024]).to(DEVICE)
spill_model = UNet(2, 1, [64, 128, 256, 512, 1024]).to(DEVICE)
nonbreaking_model = UNet(2, 1, [64, 128, 256, 512, 1024]).to(DEVICE)

plunge_model.load_state_dict(torch.load("DT_model_plunge_400_pretrained.pth"))
spill_model.load_state_dict(torch.load("DT_model_spill_400_pretrained.pth"))
nonbreaking_model.load_state_dict(torch.load("DT_model_plunge_400.pth"))

plunge_history = plunge_model.train_model(train_plunge, valid_plunge, 10, 0.0001)
# spill_history = spill_model.train_model(train_spill, valid_spill, 10, 0.0001)
# nonbreaking_history = nonbreaking_model.train_model(train_nonbreaking, valid_nonbreaking, 10, 0.0001)

# plunge_model.plot_history()
# spill_model.plot_history()
# nonbreaking_model.plot_history()

# #torch.save(model.state_dict(), f"DT_model_spill_128.pth")
# torch.save(model.state_dict(), f"DT_model_spill_400_pretrained.pth")

# #torch.save(model.state_dict(), f"DT_model_plunge_128.pth")
# torch.save(model.state_dict(), f"DT_model_plunge_400_pretrained.pth")

# torch.save(model.state_dict(), f"DT_model_nonbreaking_128.pth")
# torch.save(model.state_dict(), f"DT_model_nonbreaking_400.pth")

evaluate_model(valid_spill, spill_model, savefile=None)
evaluate_model(valid_plunge, plunge_model, savefile=None)
evaluate_model(valid_nonbreaking, nonbreaking_model, savefile=None)

# Test predictive capability
ii = 10

imp = valid_plunge_dataset.__getitem__(ii)
i1 = imp[0][0][None]
i2 = imp[0][1][None]

plt.imshow(i2[0])
plt.axis('off')
plt.show()

j = 15
for i in range(j):
    model.eval()
    with torch.no_grad():
        inp = torch.cat((i1, i2), dim=0)[None]
        i1 = i2
        i2 = plunge_model(inp.to(DEVICE))[0].to('cpu')
        plt.imshow(i2[0])
        plt.axis('off')
        plt.savefig(f"DT_predictions_plunge1_{i}")
        plt.show()

spill_model = UNet(2, 1, [64, 128, 256, 512, 1024]).to(DEVICE)
spill_model.load_state_dict(torch.load("DT_model_spill_400_pretrained.pth"))

plunge_model = UNet(2, 1, [64, 128, 256, 512, 1024]).to(DEVICE)
plunge_model.load_state_dict(torch.load("DT_model_plunge_400_pretrained.pth"))

nonbreaking_model = UNet(2, 1, [64, 128, 256, 512, 1024]).to(DEVICE)
nonbreaking_model.load_state_dict(torch.load("DT_model_nonbreaking_400.pth"))

imsi = 400
noaug = transformations.get_transform(image_shape=(imsi, imsi))
# aug = transformations.get_transform(augment = True, image_shape=(imsi, imsi))

# train_plunge_dataset = SeqImages('data/IMGS/IR/train/plunge', noaug)
# train_plunge = DataLoader(train_plunge_dataset, batch_size=1, shuffle=True, num_workers=8)
valid_plunge_dataset = SeqImages('data/IMGS/IR/valid/plunge', noaug)
valid_plunge = DataLoader(valid_plunge_dataset, batch_size=1, shuffle=True, num_workers=8)


# train_spill_dataset = SeqImages('data/IMGS/IR/train/spill', noaug)
# train_spill = DataLoader(train_spill_dataset, batch_size=1, shuffle=True, num_workers=8)
valid_spill_dataset = SeqImages('data/IMGS/IR/valid/spill', noaug)
valid_spill = DataLoader(valid_spill_dataset, batch_size=1, shuffle=True, num_workers=8)


# train_nonbreaking_dataset = SeqImages('data/IMGS/IR/train/nonbreaking', noaug)
# train_nonbreaking = DataLoader(train_nonbreaking_dataset, batch_size=1, shuffle=True, num_workers=8)
valid_nonbreaking_dataset = SeqImages('data/IMGS/IR/valid/nonbreaking', noaug)
valid_nonbreaking = DataLoader(valid_nonbreaking_dataset, batch_size=1, shuffle=True, num_workers=8)

sp, se, sa = get_model_losses(valid_spill, [1,0,0], spill_model, plunge_model, nonbreaking_model)
pp, pe, pa = get_model_losses(valid_plunge, [0,1,0], spill_model, plunge_model, nonbreaking_model)
np, ne, na = get_model_losses(valid_nonbreaking, [0,0,1], spill_model, plunge_model, nonbreaking_model)

preds = sp + pp + np
errors = se + pe + ne
actual = sa + pa + na

# train_errors = errors
# train_preds = preds
# train_actual = actual


val_errors = errors
val_preds = preds
val_actual = actual

df

train_dict = {'spill_error':train_error[:,0], 'plunge_error':train_error[:,1], 'nonbreaking_error':train_error[:,2], 'train_actual':np.argmax(train_actual, axis=1)}
df = pd.DataFrame(train_dict)
df.to_csv('DT_train_errors.csv')


val_dict = {'spill_error':val_error[:,0], 'plunge_error':val_error[:,1], 'nonbreaking_error':val_error[:,2], 'val_actual':np.argmax(val_actual, axis=1)}
df = pd.DataFrame(val_dict)
df.to_csv('DT_val_errors.csv')

df = pd.read_csv('DT_val_errors.csv')
df

classreport = classification_report(np.argmax(actual, axis=1), np.argmax(preds, axis=1))
print(classreport)

get_confusion_matrix(val_predictions, val_actual)

# mod = rfc(n_estimators=100, max_depth=5, max_leaf_nodes=5, class_weight="balanced", random_state=2020, bootstrap=True)
mod = dtc(max_depth=5, random_state=2020, max_leaf_nodes=5, class_weight="balanced")


mod.fit(train_errors, train_actual)
tree_rules = export_text(mod, feature_names=['spill', 'plunge', 'nonbreaking'])


val_predictions = mod.predict(val_errors)

classreport = classification_report(np.argmax(val_actual, axis=1), np.argmax(val_predictions, axis=1))
print(classreport)
print(tree_rules)

pre_train, pre_valid = load_data_seq_OF(128, 'nonbreaking')
train_plunge, valid_plunge = load_data_seq_OF(400, 'plunge')
train_spill, valid_spill = load_data_seq_OF(400, 'spill')
train_nonbreaking, valid_nonbreaking = load_data_seq_OF(400, 'nonbreaking')

OFmodel = UNet(5, 1, [64, 128, 256, 512, 1024]).to(DEVICE)
plunge_OFmodel = UNet(5, 1, [64, 128, 256, 512, 1024]).to(DEVICE)
spill_OFmodel = UNet(5, 1, [64, 128, 256, 512, 1024]).to(DEVICE)
nonbreaking_OFmodel = UNet(5, 1, [64, 128, 256, 512, 1024]).to(DEVICE)

OF_history = train_model(OFmodel, train_nonbreaking, valid_nonbreaking, 30, 0.01)
torch.save(OFmodel.state_dict(), f"DT_OF_model_pretrain_128.pth")
plot_history(OF_history)

spill_OFmodel.load_state_dict(torch.load("DT_model_pretrain_128.pth"))
plunge_OFmodel.load_state_dict(torch.load("DT_model_pretrain_128.pth"))
nonbreaking_OFmodel.load_state_dict(torch.load("DT_model_pretrain_128.pth"))


p_history = train_model(plunge_OFmodel, train_plunge, valid_plunge, 30, 0.001)
torch.save(plunge_OFmodel.state_dict(), f"DT_OF_model_plunge_400.pth")
plot_history(p_history)

s_history = train_model(spill_OFmodel, train_spill, valid_spill, 30, 0.001)
torch.save(spill_OFmodel.state_dict(), f"DT_OF_model_spill_400.pth")
spill_OFmodel.plot_history()

n_history = train_model(nonbreaking_OFmodel, train_nonbreaking, valid_nonbreaking, 30, 0.001)
torch.save(nonbreaking_OFmodel.state_dict(), f"DT_OF_model_nonbreaking_400.pth")
nonbreaking_OFmodel.plot_history()
