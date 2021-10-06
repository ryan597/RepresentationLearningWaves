"""
Contains useful functions for the PyTorch model, class definition for the data
pipeline, loading and for generating the results.
"""
###############################################################################

# Written by Ryan Smith
# ryan.smith@ucdconnect.ie
# github.com/ryan597/DynamicTextureWaves

###############################################################################

# Python Imports
import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

# Pytorch imports
from torch.utils.data import DataLoader, Dataset

###############################################################################
# Data Pipeline


class InputImages(Dataset):
    def __init__(self, path, transform=None):
        self.file_path = path
        self.files = glob.glob(self.file_path+'/*.png') +\
            glob.glob(self.file_path+'/*.jpg')
        self.transform = transform
        self.dataset_len = len(self.files)

    def __getitem__(self, index):
        image1 = self.fetch_image(index)
        image2 = self.fetch_image(index+1)
        image3 = self.fetch_image(index+2)

        input_images = np.array([image1, image2])

        return input_images, image3

    def __len__(self):
        return self.dataset_len

    def fetch_image(self, index):
        return cv2.imread(self.files[index], cv2.IMREAD_GRAYSCALE)


def get_transform(image_size):
    return None


def load_data(path, image_size, batch_size=1, shuffle=True):
    transform = get_transform(image_size=image_size)
    dataset = InputImages(path, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=8)
    return dataloader


def wave_sort(path, I1, wave=[]):
    wave.append(I1)
    for i in range(1, 100):
        I2 = I1 + i
        if os.path.isfile(path+str(I2)+'.jpg'):
            wave = wave_sort(path, I2, wave)
            return wave
    return wave


def show_samples(dataloader, no_samples=5):
    fig, ax = plt.subplots(no_samples,
                           3,
                           gridspec_kw={'wspace': 0, 'hspace': 0},
                           subplot_kw={'xticks': [], 'yticks': []})

    for i, (samples, truth) in enumerate(dataloader):
        ax[i, 0].imshow(samples[0, 0])
        ax[i, 1].imshow(samples[0, 1])
        ax[i, 2].imshow(truth[0])

        if i == (no_samples - 1):
            break
    fig.suptitle("Sample images from dataset")
    # fig.supxlabel("1st, 2nd and 3rd Image from Sequence")
    # fig.supylabel("Samples")
    plt.show()


###############################################################################
# Training loop functions

"""
def evaluate_model(valid, model, savefile=None):
    with torch.no_grad():
        for i, (batch_pair, batch_nxt) in enumerate(valid):
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5,
                                                          figsize=(15, 15))
            for j, img_pair in enumerate(batch_pair):
                ax1.imshow(img_pair[0])
                ax1.axis('off')
                ax2.imshow(img_pair[1])
                ax2.axis('off')
                ax3.imshow(batch_nxt[0][0])
                ax3.axis('off')
                pr = model(batch_pair.to(DEVICE))[0][0].detach()\
                    .to('cpu').numpy()
                pm = (pr - np.min(pr))/(np.max(pr)-np.min(pr))
                ax4.imshow(pm)
                ax4.axis('off')
                bnxt = ((batch_nxt[0][0].numpy() -
                        np.min(batch_nxt[0][0].numpy())) /
                        (np.max(batch_nxt[0][0].numpy()) -
                        np.min(batch_nxt[0][0].numpy())))
                ax5.imshow(1 - np.abs(img_pair[1] - bnxt), cmap='gray')
                ax5.axis('off')
                if savefile is not None:
                    plt.savefig(f"{savefile}_{i}")
                fig.show()
                break
            if i == 4:
                break
"""