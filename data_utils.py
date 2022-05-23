"""
Contains useful functions for the PyTorch model, class definition for the data
pipeline, loading and for generating the results.
"""

# Python Imports
from os.path import exists
import cv2
import glob
import random
import matplotlib.pyplot as plt

# Pytorch imports
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

###############################################################################
# Data Pipeline


class InputSequence(Dataset):
    def __init__(self, path, image_shape, masks=False, input_N=2):
        self.image_shape = image_shape
        self.folder_path = path
        self.input_N = input_N
        self.folders = glob.glob(self.folder_path + "/wave_*")
        if masks:
            self.sequences = self.generate_sequences_masks()
        else:
            self.sequences = self.generate_sequences()
        self.dataset_len = len(self.sequences)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        p1, p2, p3 = self.sequences[index]
        image1 = self.fetch_image(p1)
        image2 = self.fetch_image(p2)
        image3 = self.fetch_image(p3)
        image1, image2, image3 = self.get_transform(image1, image2, image3)
        if self.input_N == 2:
            input_images = torch.cat((image1, image2), dim=0)
        else:
            input_images = torch.cat((image2, image2, image2), dim=0)
        return (input_images, image3)

    def fetch_image(self, path):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    def generate_sequences(self):
        sequences = {}
        counter = 0
        for folder in self.folders:
            files = glob.glob(f"{folder}/*.png") +\
                glob.glob(f"{folder}/*.jpg")
            for (img1, img2, img3) in zip(files[:-2], files[1:-1], files[2:]):
                sequences[counter] = (img1, img2, img3)
                counter += 1

        return sequences

    def generate_sequences_masks(self):
        sequences = {}
        counter = 0
        for folder in self.folders:
            files = glob.glob(f"{folder}/*.png") +\
                glob.glob(f"{folder}/*.jpg")
            for (img1, img2, img3) in zip(files[:-2], files[1:-1], files[2:]):
                img3 = "data/segmentation/BW/" + img3[-9:]
                if exists(img3):
                    sequences[counter] = (img1, img2, img3)
                    counter += 1
        return sequences

    def get_transform(self, *args):
        images = []
        hflip = random.random()
        i, j, h, w = T.RandomCrop.get_params(
            TF.to_tensor(args[0]), output_size=self.image_shape)
        d = T.RandomRotation.get_params(degrees=[-45, 45])
        for image in args:
            # Transform to tensor
            image = TF.to_tensor(image)
            # Resize
            resize = T.Resize(size=self.image_shape)
            image = resize(image)
            # Random crop
            image = TF.crop(image, i, j, h, w)
            # Random horizontal flipping
            if hflip > 0.5:
                image = TF.hflip(image)
            # Random Rotation
            image = TF.rotate(image, angle=d)

            images.append(image)
        return images


def load_data(path, image_shape,
              batch_size=16, shuffle=True, **kwargs):
    dataset = InputSequence(path, image_shape, kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=80, persistent_workers=True)
    return dataloader


def show_samples(dataloader, num_samples=5):
    fig, ax = plt.subplots(num_samples,
                           3,
                           gridspec_kw={'wspace': 0, 'hspace': 0},
                           subplot_kw={'xticks': [], 'yticks': []})

    for i, (samples, truth) in enumerate(dataloader):
        # enumerate delivers a batch, just pick the first in the batch
        ax[i, 0].imshow(samples[0][0].numpy())  # first channel
        ax[i, 1].imshow(samples[0][1].numpy())  # second channel
        ax[i, 2].imshow(truth[0][0].numpy())

        if i == (num_samples - 1):
            break
    fig.suptitle("Sample images from dataset")
    # fig.supxlabel("1st, 2nd and 3rd Image from Sequence")
    # fig.supylabel("Samples")
    plt.show()
