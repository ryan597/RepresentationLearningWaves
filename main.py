import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import utils
from models import my_CNNs

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

################################################################################
# Data Pipeline

################################################################################
# Loading Models

################################################################################
# Training

################################################################################


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, *args, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, *args, **kwargs),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, layer_sizes, *args, **kwargs):
        super().__init__()
        down_sizes = [in_channels, *layer_sizes]
        up_sizes = [*layer_sizes[::-1] , out_channels]

        self.conv_down = nn.ModuleList([
            DoubleConv(in_ch, out_ch) for in_ch, out_ch in zip(down_sizes, down_sizes[1:])
        ])

        self.down_sample = nn.ModuleList([
            nn.MaxPool2d(kernel_size=2, stride=2) for i in range(len(layer_sizes)-1)
        ])

        self.up_sample = nn.ModuleList([
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2) for in_ch, out_ch in zip(up_sizes, up_sizes[1:-1])
        ])

        self.conv_up = nn.ModuleList([
            DoubleConv(in_ch, out_ch) for in_ch, out_ch in zip(up_sizes, up_sizes[1:])
        ])


    def forward(self, x):
        copies = []
        for i, conv in enumerate(self.conv_down[:-1]):
            x = conv(x)
            copies.append(x)
            x = self.down_sample[i](x)
        
        x = self.conv_down[-1](x)
        for i, conv in enumerate(self.conv_up[:-1]):
            x = self.up_sample[i](x)
            x = conv(torch.cat((x, copies[-(i+1)]), dim=1))

        x = self.conv_up[-1](x)
        return x

def train_model(model, train, valid, epochs, learning_rate, verbose=1):     
    history = {"loss":[], "val_loss":[], "epoch":[], "lr":[]}

    criterion = nn.L1Loss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*3, eta_min=1e-7)

    for i in range(epochs):
        history["epoch"].append(i+1)
        history["lr"].append(scheduler.get_lr())
        accum_loss = 0
        total_loss = 0
        optimizer.zero_grad()
        print(f"learning rate {scheduler.get_lr()}")

        for j, (inputs, nxt) in enumerate(train):
            inputs = inputs.to(DEVICE)
            nxt = nxt.to(DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, nxt)
            accum_loss += loss.item()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            if j % 10 == 0 and verbose:
                print(f"loss \t {accum_loss / 10}")
                total_loss += accum_loss
                accum_loss = 0

        optimizer.step()
        optimizer.zero_grad()
        total_loss *= 1 / len(train)
        history["loss"].append(total_loss)
        
        print(f"Epoch \t {i+1} finished")
        scheduler.step()
        # Validation
        with torch.no_grad():
            valid_loss = 0
            for j, (inputs, nxt) in enumerate(valid):

                inputs = inputs.to(DEVICE)
                nxt = nxt.to(DEVICE) 

                outputs = model(inputs)
                val_loss = criterion(outputs, nxt)
                valid_loss += val_loss.item()
            
            valid_loss *= 1/len(valid)
            history["val_loss"].append(valid)
            print(f"validation loss : \t{valid_loss} \n")

    return history

def plot_history(history):
    plt.plot(history["epoch"], history["loss"], label="training loss")
    plt.plot(history["epoch"], history["val_loss"], label="validation loss")
    plt.title('Training and Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

"""## Dynamic Texture prediction"""

def wave_sort(path, I1, wave=[]):
    wave.append(I1)
    for i in range(1, 100):
        I2 = I1 + i
        if os.path.isfile(path+str(I2)+'.jpg'):
            wave = wave_sort(path, I2, wave)
            return wave
    return wave


class SeqImages(Dataset):
    def __init__(self, path, transform):
        self.file_path = path
        files = glob.glob(self.file_path+'/*.jpg')
        self.files = map(lambda x: int(x.split('/')[-1][:-4]), files)
        self.transform = transform

        self.waves = None
        self.dataset_len = len(files)
        self.wave_dict = {}


    def __getitem__(self, index):
        if self.waves is None:
            waves = []
            for i in list(self.files):
                try:
                    if i not in waves[-1]:
                        waves.append( wave_sort(self.file_path+'/', i, []) )
                except:
                    waves.append( wave_sort(self.file_path+'/', i, []) )
            self.waves = waves
            for i in range(0, len(waves)):
                self.wave_dict[i] = len(waves[i])

        count = 0
        for i in range(0, len(self.wave_dict)):
            if index >= count:
                sample_wave = self.waves[i]       
                count += self.wave_dict[i]
        
        # check edge conditions
        rand_indx = index - count + len(sample_wave)
        if rand_indx == len(sample_wave):
            rand_indx -= 3
        elif rand_indx == len(sample_wave) - 1:
            rand_indx -= 2
        elif rand_indx == len(sample_wave) - 2:
            rand_indx -= 1
        
        img_one = Image.open(self.file_path+'/'+str(sample_wave[rand_indx])+'.jpg')
        img_two = Image.open(self.file_path+'/'+str(sample_wave[rand_indx+1])+'.jpg')
        img_thr = Image.open(self.file_path+'/'+str(sample_wave[rand_indx+2])+'.jpg')

        img_pair = torch.stack((self.transform(img_one), self.transform(img_two)))

        return img_pair, self.transform(img_thr)
        
    def __len__(self):
        return self.dataset_len


def load_data_seq(image_size, label, batch_size=1, shuffle=True):
    noaug = transformations.get_transform(image_shape=(image_size, image_size))

    train_dataset = SeqImages(f'data/IMGS/IR/train/{label}', noaug)
    train = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)
    valid_dataset = SeqImages(f'data/IMGS/IR/valid/{label}', noaug)
    valid = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=8)

    return train, valid

imsize = 400
noaug = transformations.get_transform(image_shape=(imsize, imsize))
#aug = transformations.get_transform(augment = True, image_shape=(imsize, imsize))

#pre_train, pre_valid = load_data_seq(128, 'nonbreaking')
train_plunge, valid_plunge = load_data_seq(400, 'plunge')
train_spill, valid_spill = load_data_seq(400, 'spill')
train_nonbreaking, valid_nonbreaking = load_data_seq(400, 'nonbreaking')

plunge_model = UNet(2, 1, [64, 128, 256, 512, 1024]).to(DEVICE)
spill_model = UNet(2, 1, [64, 128, 256, 512, 1024]).to(DEVICE)
nonbreaking_model = UNet(2, 1, [64, 128, 256, 512, 1024]).to(DEVICE)

plunge_model.load_state_dict(torch.load("DT_model_plunge_400_pretrained.pth"))
spill_model.load_state_dict(torch.load("DT_model_spill_400_pretrained.pth"))
nonbreaking_model.load_state_dict(torch.load("DT_model_plunge_400.pth"))

plunge_history = plunge_model.train_model(train_plunge, valid_plunge, 10, 0.0001)
#spill_history = spill_model.train_model(train_spill, valid_spill, 10, 0.0001)
#nonbreaking_history = nonbreaking_model.train_model(train_nonbreaking, valid_nonbreaking, 10, 0.0001)

#plunge_model.plot_history()
#spill_model.plot_history()
#nonbreaking_model.plot_history()

##torch.save(model.state_dict(), f"DT_model_spill_128.pth")
#torch.save(model.state_dict(), f"DT_model_spill_400_pretrained.pth")

##torch.save(model.state_dict(), f"DT_model_plunge_128.pth")
#torch.save(model.state_dict(), f"DT_model_plunge_400_pretrained.pth")

#torch.save(model.state_dict(), f"DT_model_nonbreaking_128.pth")
#torch.save(model.state_dict(), f"DT_model_nonbreaking_400.pth")

def evaluate_model(valid, model, savefile=None):
    with torch.no_grad():
        for i, (batch_pair, batch_nxt) in enumerate(valid):
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(15,15))
            for j, img_pair in enumerate(batch_pair):
                ax1.imshow(img_pair[0])
                ax1.axis('off')
                ax2.imshow(img_pair[1])
                ax2.axis('off')
                ax3.imshow(batch_nxt[0][0])
                ax3.axis('off')
                pr = model(batch_pair.to(DEVICE))[0][0].detach().to('cpu').numpy()
                pm = (pr - np.min(pr))/(np.max(pr)-np.min(pr))
                ax4.imshow(pm)
                ax4.axis('off')
                bnxt = (batch_nxt[0][0].numpy()-np.min(batch_nxt[0][0].numpy()))/(np.max(batch_nxt[0][0].numpy())-np.min(batch_nxt[0][0].numpy()))
                #bnxt = (img_pair[1].numpy()-np.min(img_pair[1].numpy()))/(np.max(img_pair[1].numpy())-np.min(img_pair[1].numpy()))
                ax5.imshow(1 - np.abs(img_pair[1] - bnxt), cmap='gray')
                ax5.axis('off')
                if savefile != None:
                    plt.savefig(f"{savefile}_{i}")  # plunge_model.plot_history()
                fig.show()
                break
            if i == 4:
                break

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
#aug = transformations.get_transform(augment = True, image_shape=(imsi, imsi))

#train_plunge_dataset = SeqImages('data/IMGS/IR/train/plunge', noaug)
#train_plunge = DataLoader(train_plunge_dataset, batch_size=1, shuffle=True, num_workers=8)
valid_plunge_dataset = SeqImages('data/IMGS/IR/valid/plunge', noaug)
valid_plunge = DataLoader(valid_plunge_dataset, batch_size=1, shuffle=True, num_workers=8)


#train_spill_dataset = SeqImages('data/IMGS/IR/train/spill', noaug)
#train_spill = DataLoader(train_spill_dataset, batch_size=1, shuffle=True, num_workers=8)
valid_spill_dataset = SeqImages('data/IMGS/IR/valid/spill', noaug)
valid_spill = DataLoader(valid_spill_dataset, batch_size=1, shuffle=True, num_workers=8)


#train_nonbreaking_dataset = SeqImages('data/IMGS/IR/train/nonbreaking', noaug)
#train_nonbreaking = DataLoader(train_nonbreaking_dataset, batch_size=1, shuffle=True, num_workers=8)
valid_nonbreaking_dataset = SeqImages('data/IMGS/IR/valid/nonbreaking', noaug)
valid_nonbreaking = DataLoader(valid_nonbreaking_dataset, batch_size=1, shuffle=True, num_workers=8)

def get_model_losses(valid, label, spill_model, plunge_model, nonbreaking_model):
    preds = []
    errors = []
    actual = []
    with torch.no_grad():
        for j, (pair, nxt) in enumerate(valid):
            criterion = nn.L1Loss().to(DEVICE)
            
            pair = pair.to(DEVICE)
            nxt = nxt.to(DEVICE) 

            spill_output = spill_model(pair)
            plunge_output = plunge_model(pair)
            nonbreaking_output = nonbreaking_model(pair)
            spill_loss = criterion(spill_output, nxt).to('cpu')
            plunge_loss = criterion(plunge_output, nxt).to('cpu')
            nonbreaking_loss = criterion(nonbreaking_output, nxt).to('cpu')

            batch_preds = [spill_loss, plunge_loss, nonbreaking_loss]
            batch_preds_hot = (batch_preds == np.min(batch_preds)).astype(int)

            errors.append(batch_preds)
            preds.append(batch_preds_hot)
            actual.append(label)

    return preds, errors, actual


sp, se, sa = get_model_losses(valid_spill, [1,0,0], spill_model, plunge_model, nonbreaking_model)
pp, pe, pa = get_model_losses(valid_plunge, [0,1,0], spill_model, plunge_model, nonbreaking_model)
np, ne, na = get_model_losses(valid_nonbreaking, [0,0,1], spill_model, plunge_model, nonbreaking_model)

preds = sp + pp + np
errors = se + pe + ne
actual = sa + pa + na

#train_errors = errors
#train_preds = preds
#train_actual = actual


val_errors = errors
val_preds = preds
val_actual = actual

df

import pandas as pd

train_dict = {'spill_error':train_error[:,0], 'plunge_error':train_error[:,1], 'nonbreaking_error':train_error[:,2], 'train_actual':np.argmax(train_actual, axis=1)}
df = pd.DataFrame(train_dict)
df.to_csv('DT_train_errors.csv')


val_dict = {'spill_error':val_error[:,0], 'plunge_error':val_error[:,1], 'nonbreaking_error':val_error[:,2], 'val_actual':np.argmax(val_actual, axis=1)}
df = pd.DataFrame(val_dict)
df.to_csv('DT_val_errors.csv')

import pandas as pd
df = pd.read_csv('DT_val_errors.csv')
df

classreport = classification_report(np.argmax(actual, axis=1), np.argmax(preds, axis=1))
print(classreport)

import seaborn as sns

def get_confusion_matrix(val_predictions, val_actual):
    labels = np.argmax(val_actual, axis=1)
    pre = np.argmax(val_predictions, axis=1)

    cm = confusion_matrix(labels, pre)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.set(font_scale=2)
    sns.heatmap(cm,
                annot=True,
                cmap = sns.cubehelix_palette(dark=0, light=1, as_cmap=True), cbar=False) 

    classes = ["spill", "plunge", "nonbreaking"]
    yclasses = ['true '+t for t in classes]
    tick_marks = np.arange(len(classes))+.5
    plt.xticks(tick_marks, classes, rotation=0,fontsize=10)
    plt.yticks(tick_marks, yclasses, rotation=45, fontsize=10)
    plt.show()


get_confusion_matrix(val_predictions, val_actual)

from sklearn.linear_model import LogisticRegression as lr
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.tree import export_text
#from sklearn.ensemble import RandomForestClassifier as rfc


#mod = rfc(n_estimators=100, max_depth=5, max_leaf_nodes=5, class_weight="balanced", random_state=2020, bootstrap=True)
mod = dtc(max_depth=5, random_state=2020, max_leaf_nodes=5, class_weight="balanced")


mod.fit(train_errors, train_actual)
tree_rules = export_text(mod, feature_names=['spill', 'plunge', 'nonbreaking'])


val_predictions = mod.predict(val_errors)

classreport = classification_report(np.argmax(val_actual, axis=1), np.argmax(val_predictions, axis=1))
print(classreport)
print(tree_rules)

"""## Dynamic Texture prediction & OF"""

def wave_sort(path, I1, wave=[]):
    wave.append(I1)
    i = 100
    I2 = I1 + i
    if os.path.isfile(path+str(I2)+'.png'):
        wave = wave_sort(path, I2, wave)
        return wave
    return wave

class SeqPairOFImages(Dataset):
    def __init__(self, ir_path, transform):
        self.ir_file_path = ir_path
        self.of_file_path = ir_path[:10]+'OF'+ir_path[12:]
        of_files = glob.glob(self.of_file_path+'/*.png')

        self.of_files = map(lambda x: int(x.split('/')[-1][:-4]), of_files)
        self.transform = transform

        self.waves = None
        self.dataset_len = len(of_files)
        self.wave_dict = {}


    def __getitem__(self, index):
        if self.waves is None:
            waves = []
            for i in list(self.of_files):
                try:
                    if i not in waves[-1]:
                        waves.append( wave_sort(self.of_file_path+'/', i, []) )
                except:
                    waves.append( wave_sort(self.of_file_path+'/', i, []) )
            self.waves = waves
            for i in range(0, len(waves)):
                self.wave_dict[i] = len(waves[i])

        count = 0
        for i in range(0, len(self.wave_dict)):
            if index >= count:
                sample_wave = self.waves[i]       
                count += self.wave_dict[i]
        
        # check edge conditions
        rand_indx = count - index
        if rand_indx >= len(sample_wave):
            rand_indx = len(sample_wave) - 3
        elif rand_indx == len(sample_wave) - 1:
            rand_indx -= 2
        elif rand_indx == len(sample_wave) - 2:
            rand_indx -= 1
        

        img_one = Image.open(self.ir_file_path+'/'+str(sample_wave[rand_indx])+'.jpg')
        img_two = Image.open(self.ir_file_path+'/'+str(sample_wave[rand_indx+1])+'.jpg')
        img_thr = Image.open(self.ir_file_path+'/'+str(sample_wave[rand_indx+2])+'.jpg')

        img_of = Image.open(self.of_file_path+'/'+str(sample_wave[rand_indx])+'.png')

        img_inputs = torch.stack((self.transform(img_one),
                                self.transform(img_two),
                                self.transform(img_of)))
        return img_inputs, self.transform(img_thr)
        
    def __len__(self):
        return self.dataset_len

def load_data_seq_OF(image_size, label, batch_size=1, shuffle=True):
    noaug = transformations.get_transform(image_shape=(image_size, image_size))

    train_dataset = SeqPairOFImages(f'data/IMGS/IR/train/{label}', noaug)
    train = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)
    valid_dataset = SeqPairOFImages(f'data/IMGS/IR/test/{label}', noaug)
    valid = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=8)

    return train, valid

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

"""## Dyanmic Texture prediction & Optical Flow late"""



