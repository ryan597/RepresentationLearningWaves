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
    def __init__(self, path, image_shape, masks=False,
                 seq_length=2, step=1, aug=False, channels=1):
        self.image_shape = image_shape
        self.folder_path = path
        self.masks = masks
        self.seq_length = seq_length
        self.aug = aug
        self.channels = channels
        self.step = step
        self.folders = glob.glob("v*", root_dir=self.folder_path)
        if masks:
            self.sequences = self.generate_masks()
        else:
            self.sequences = self.generate_sequences()
        self.dataset_len = len(self.sequences)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        p1, p2, p3, p4, p5 = self.sequences[index]
        img5 = self.fetch_image(p5)
        img4 = self.fetch_image(p4)
        img3 = self.fetch_image(p3)
        img2 = self.fetch_image(p2)
        img1 = self.fetch_image(p1)
        img1, img2, img3, img4, img5 = self.transform(img1, img2, img3, img4, img5)
        normal = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

        if self.seq_length == 2:
            if self.channels == 3:
                input_images = torch.stack((img4, img4, img4), dim=0)
                input_images = normal(input_images)
            else:
                input_images = img4[None]  # add channel dimension

        if self.seq_length == 3:
            if self.channels == 3:  # Deal with pretrained RGB models
                img3 = torch.stack((img3, img3, img3), dim=0)
                img4 = torch.stack((img4, img4, img4), dim=0)
                # Normalise for pretrained models
                img3 = normal(img3)
                img4 = normal(img4)
                input_images = torch.cat((img3, img4), dim=0)
            else:
                # add channel dim and cat along channel
                input_images = torch.cat((img3[None], img4[None]), dim=0)

        if self.seq_length == 5:
            input_images = torch.cat((img1[None], img2[None],
                img3[None], img4[None]), dim=0)

        if self.masks:
            img5 = torch.stack((img5, 1 - img5), dim=0)
        else:
            img5 = img5[None]  # Add a channel dimension

        return (input_images, img5)

    def fetch_image(self, path):
        return cv2.imread(self.folder_path + "/" + path, cv2.IMREAD_GRAYSCALE)

    def check_seq(self, *args, step=1):
        # ensure all images are same timestep apart
        for img1, img2 in zip(args, args[1:]):
            if (int(img2[3:-4]) - int(img1[3:-4]) != step):
                return False
        return True

    def generate_sequences(self):
        # Generate sequence of 5 images with same timestep,
        # if only using a seq of length 3, we simply ignore images 1 & 2 later.
        # Avoids extra complications in ensuring the sequences tested on are
        # the same
        sequences = {}
        counter = 0
        for folder in self.folders:
            files = glob.glob(f"{folder}/*.png", root_dir=self.folder_path)
            for n in range(self.step):
                f_temp = sorted(files)[n::self.step]  # take every Nth (N=step) element
                for (img1, img2, img3, img4, img5) in \
                  zip(f_temp, f_temp[1:], f_temp[2:], f_temp[3:], f_temp[4:]):
                    if self.check_seq(img1, img2, img3, img4, img5, step=self.step):
                        sequences[counter] = (img1, img2, img3, img4, img5)
                        counter += 1
        return sequences

    def generate_masks(self):
        sequences = {}
        counter = 0
        for folder in self.folders:
            files = glob.glob(f"{folder}/*.png", root_dir=self.folder_path)
            for n in range(self.step):
                f_temp = sorted(files)[n::self.step]  # take every Nth (N=step) element
                for (img1, img2, img3, img4, img5) in \
                  zip(f_temp, f_temp[1:], f_temp[2:], f_temp[3:], f_temp[4:]):
                    if self.check_seq(img1, img2, img3, img4, img5, step=self.step):
                        img5 = "/masks/BW/" + img4[3:]  # get mask of img4
                    if exists(self.folder_path + img5):
                        sequences[counter] = (img1, img2, img3, img4, img5)
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
              masks=False, seq_length=3, step=1, aug=False, channels=1):
    dataset = InputSequence(path, image_shape, masks, seq_length, step, aug, channels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=4, persistent_workers=True,
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
