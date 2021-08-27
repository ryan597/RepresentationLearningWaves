################################################################################

# Written by Ryan Smith
# ryan.smith@ucdconnect.ie

################################################################################

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

################################################################################


def wave_sort(path, I1, wave=[]):
    wave.append(I1)
    for i in range(1, 100):
        I2 = I1 + i
        if os.path.isfile(path+str(I2)+'.jpg'):
            wave = wave_sort(path, I2, wave)
            return wave
    return wave


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


def plot_history(history):
    plt.plot(history["epoch"], history["loss"], label="training loss")
    plt.plot(history["epoch"], history["val_loss"], label="validation loss")
    plt.title('Training and Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


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


def get_confusion_matrix(val_predictions, val_actual):
    labels = np.argmax(val_actual, axis=1)
    pre = np.argmax(val_predictions, axis=1)

    cm = confusion_matrix(labels, pre)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.set(font_scale=2)
    sns.heatmap(cm,
                annot=True,
                cmap=sns.cubehelix_palette(dark=0, light=1, as_cmap=True), cbar=False)

    classes = ["spill", "plunge", "nonbreaking"]
    yclasses = ['true '+t for t in classes]
    tick_marks = np.arange(len(classes))+.5
    plt.xticks(tick_marks, classes, rotation=0, fontsize=10)
    plt.yticks(tick_marks, yclasses, rotation=45, fontsize=10)
    plt.show()