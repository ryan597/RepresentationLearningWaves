"""
Contains useful functions for the PyTorch model, class definition for the data
pipeline, loading and for generating the results.
"""

from os.path import exists
import cv2
import glob
import random
# import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class InputSequence(Dataset):
    def __init__(self, path, image_shape, masks=False, dual=False, aug=False):
        self.image_shape = image_shape
        self.folder_path = path
        self.masks = masks
        self.dual = dual
        self.aug = aug
        self.folders = glob.glob("v*", root_dir=self.folder_path)
        if masks:
            self.sequences = self.generate_masks()
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
        image1, image2, image3 = self.transform(image1, image2, image3)
        #normal = T.Normalize(mean=[0.485, 0.456, 0.406],
        #                     std=[0.229, 0.224, 0.225])

        if self.dual:  # 6 channels
            #image1 = normal(torch.stack((image1, image1, image1), dim=0))
            image1 = torch.stack((image1, image1, image1), dim=0)
            #image2 = normal(torch.stack((image2, image2, image2), dim=0))
            image2 = torch.stack((image2, image2, image2), dim=0)
            input_images = torch.cat((image1, image2), dim=0)
        else:  # 3 channels
            input_images = torch.stack((image2, image2, image2), dim=0)
            #input_images = normal(input_images)
        if self.masks:
            image3 = torch.stack((image3, 1 - image3), dim=0)
        else:
            image3 = image3[None]  # Add a channel dimension
        return (input_images, image3)

    def fetch_image(self, path):
        return cv2.imread(self.folder_path + "/" + path, cv2.IMREAD_GRAYSCALE)

    def check_seq(self, p1, p2, p3):
        if (int(p2[3:-4]) - int(p1[3:-4]) == 1 and
           int(p3[3:-4]) - int(p2[3:-4]) == 1):
            return True
        else:
            return False

    def generate_sequences(self):
        sequences = {}
        counter = 0
        for folder in self.folders:
            files = glob.glob(f"{folder}/*.png", root_dir=self.folder_path)
            files = sorted(files)
            for (img1, img2, img3) in zip(files[:-2], files[1:-1], files[2:]):
                if self.check_seq(img1, img2, img3):
                    sequences[counter] = (img1, img2, img3)
                    counter += 1
        return sequences

    def generate_masks(self):
        sequences = {}
        counter = 0
        for folder in self.folders:
            files = glob.glob(f"{folder}/*.png", root_dir=self.folder_path)
            files = sorted(files)
            for (img1, img2, img3) in zip(files[:-2], files[1:-1], files[2:]):
                if self.check_seq(img1, img2, img3):
                    img3 = "/masks/BW/" + img2[3:]  # get mask of img2
                    if exists(self.folder_path + img3):
                        sequences[counter] = (img1, img2, img3)
                        counter += 1
        return sequences

    def transform(self, *args):
        images = []
        if not self.aug:
            transform = T.ToTensor()
            resize = T.Resize(size=self.image_shape)
            for image in args:
                images.append(resize(transform(image))[0])
            return images

        hflip = random.random()
        vflip = random.random()
        i, j, h, w = T.RandomCrop.get_params(
            TF.to_tensor(args[0]), self.image_shape)
        d = T.RandomRotation.get_params(degrees=[-30, 30])
        for image in args:
            # Transform to tensor
            image = TF.to_tensor(image)
            # Random crop
            image = TF.crop(image, i, j, h, w)
            # Random horizontal flip
            if hflip > 0.5:
                image = TF.hflip(image)
            # Random vertical flip
            if vflip > 0.5:
                image = TF.vflip(image)
            # Random Rotation
            image = TF.rotate(image, angle=d)
            # Resize
            resize = T.Resize(size=self.image_shape)
            image = resize(image)
            images.append(image[0])  # tensors have been formated to [1, x, y]
        return images


def load_data(path, image_shape, batch_size=10, shuffle=True,
              masks=False, dual=False, aug=False):
    dataset = InputSequence(path, image_shape, masks, dual, aug)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=40, persistent_workers=True,
                            pin_memory=True)
    return dataloader


# def show_samples(dataloader, num_samples=5):
#     fig, ax = plt.subplots(num_samples,
#                            3,
#                            gridspec_kw={'wspace': 0, 'hspace': 0},
#                            subplot_kw={'xticks': [], 'yticks': []})
#
#     for i, (samples, truth) in enumerate(dataloader):
#         # enumerate delivers a batch, just pick the first in the batch
#         ax[i, 0].imshow(samples[0][0].numpy())  # first channel
#         ax[i, 1].imshow(samples[0][1].numpy())  # second channel
#         ax[i, 2].imshow(truth[0][0].numpy())
#
#         if i == (num_samples - 1):
#             break
#     fig.suptitle("Sample images from dataset")
#     # fig.supxlabel("1st, 2nd and 3rd Image from Sequence")
#     # fig.supylabel("Samples")
#     plt.show()
