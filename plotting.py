import numpy as np
import matplotlib.pyplot as plt
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename")
args = parser.parse_args()

with open(f"outputs/results/training/{args.filename}") as f:
    data_dict = json.load(f)

loss = data_dict['loss']
batch_loss = data_dict['batch_loss']
val_loss = data_dict['val_loss']
epoch = data_dict['epoch']


fig, axs = plt.subplots(1, 2, sharey=True)

axs[0].plot(epoch, loss, label="Training")

if len(val_loss) != 0:
    try:
        axs[0].plot(epoch, val_loss, label="Validation")
    except:
        axs[0].plot(epoch[:-1], val_loss, label="Validation")

axs[1].plot(np.linspace(0, epoch[-1], len(batch_loss)), batch_loss, label="Batch Loss")

axs[0].grid()
axs[1].grid()
axs[0].legend()
axs[1].legend()

axs[0].set_xlabel('Epoch')
axs[1].set_xlabel('Epoch')
axs[0].set_ylabel('L1 Loss')

axs[0].set_title('Loss')
axs[1].set_title('Batch Loss')
plt.show()


