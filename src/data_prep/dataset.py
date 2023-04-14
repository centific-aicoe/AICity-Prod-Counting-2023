import os.path

import cv2
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F


class ProductSegmentationDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, transform):
        # store image and file paths, transformations
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transform = transform

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imagePaths)

    def __getitem__(self, idx):
        # grab the image path from the current index
        imagePath = self.imagePaths[idx]

        # load the image from disk, swap its channels from BGR to RGB,
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read mask in gray scale mode
        mask = cv2.imread(self.maskPaths[idx], cv2.IMREAD_GRAYSCALE)

        # check to see if we are applying any transformations
        if self.transform is not None:
            # apply the transformations to both image and its mask
            image = self.transform(image)
            mask = self.transform(mask)

        # return a tuple of the image and its mask
        return image, mask


class ProductClassifierDataset(Dataset):
    def __init__(self, imagePaths, transform, num_classes):
        self.imagePaths = imagePaths
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imagePaths)

    def __getitem__(self, idx):
        # grab the image path from the current index
        imagePath = self.imagePaths[idx]

        # load the image from disk, swap its channels from BGR to RGB,
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # check to see if we are applying any transformations
        if self.transform is not None:
            # apply the transformations to both image and its mask
            image = self.transform(image)

        # Create y label as one hot encoder
        filename = os.path.basename(imagePath)
        class_name = filename.split("_")[0]
        class_number = int(class_name.lstrip("0"))
        y = torch.tensor(class_number-1, dtype=torch.long)
        # y = F.one_hot(x, num_classes=self.num_classes, dtype=torch.long)

        # return a tuple of the image and its y label one hot encoded
        return image, y
