"""
Contains useful functions for the PyTorch model, class definition for the data
pipeline, loading and for generating the results.
"""

# Python Imports
import cv2
import glob
import matplotlib.pyplot as plt

# Pytorch imports
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

###############################################################################
# Data Pipeline


class InputSequence(Dataset):
    def __init__(self, path, image_size):
        self.folder_path = path
        self.image_size = image_size
        self.transform = self.get_transform()
        self.folders = glob.glob(self.folder_path)
        self.sequences = self.generate_sequences()
        self.dataset_len = len(self.sequences)

    def generate_sequences(self):
        sequences = {}
        counter = 0
        for folder in self.folders:
            files = glob.glob(self.folder_path+f"/{folder}/*.png") +\
                glob.glob(self.folder_path+f"/{folder}/*.jpg")
            for (img1, img2, img3) in zip(files[:-2], files[1:-1], files[2:]):
                sequences[counter] = (img1, img2, img3)
                counter += 1

        return sequences

    def __getitem__(self, index):
        p1, p2, p3 = self.sequences[index]
        image1 = self.fetch_image(p1)
        image2 = self.fetch_image(p2)
        image3 = self.fetch_image(p3)
        input_images = torch.Tensor(image1, image2)
        return (input_images, image3)

    def __len__(self):
        return self.dataset_len

    def fetch_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return self.transform(image)

    def get_transform(self):
        transform = T.Compose(
            # T.ColorJitter(brightness, contrast, saturation, hue)
            # T.RandomAffine(degrees, translate, scale, interpolation)
            # T.RandomResizedCrop(size)
            # T.RandomRotation(degrees)
            # T.Normalize(mean, std)
            T.Resize(size=self.image_size),
            T.ToTensor()
        )
        return transform


def load_data(path, image_size=(256, 256), batch_size=1, shuffle=True):
    dataset = InputSequence(path, image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=8)
    return dataloader


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
