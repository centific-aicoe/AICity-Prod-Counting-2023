
import os
import sys
import random
import numpy as np

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms

# Adding working directory to path to run the files as standalone
curr_wd = os.getcwd()
if curr_wd not in sys.path:
    sys.path.append(curr_wd)  # add working directory to PATH

from src.utils.helpers import get_mask_file_names
from src.data_prep.dataset import ProductSegmentationDataset


class SegDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size

    # Use the prepare_data if you are using any torch datasets for downloading
    def prepare_data(self):
        print("Preparing data module")
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        # Run the custom dataset logic to create datasets
        # Create train, valid and test datasets
        # train_dataset, val_dataset, test_dataset = prepare_seg_datasets(data_dir=self.data_dir)

        images_dir = os.path.join(self.data_dir, "train")
        masks_dir = os.path.join(self.data_dir, "segmentation_labels")

        # Logic to do a stratified split of images across all the sets
        # 80% Train, 10% val and 10% Test
        num_classes = 116

        # Initialize empty lists for train valid and test sets
        lst_train_images, lst_valid_images, lst_test_images = [], [], []
        lst_train_masks, lst_valid_masks, lst_test_masks = [], [], []

        # Loop over each class
        for i in range(1, num_classes + 1):
            print("Preparing data for class: ", str(i))

            # get all the files for a specific class
            lst_tmp_spec_class = [item for item in os.listdir(images_dir) if item.startswith(str(i).zfill(5))]

            # train
            num_samples_train = int(0.8 * len(lst_tmp_spec_class))
            num_samples_valid = int(0.1 * len(lst_tmp_spec_class))
            num_samples_test = len(lst_tmp_spec_class) - num_samples_valid - num_samples_train

            # shuffle the list
            random.Random(4).shuffle(lst_tmp_spec_class)

            # Getting specific lists for each set
            lst_train_images_class = lst_tmp_spec_class[:num_samples_train]
            lst_valid_images_class = lst_tmp_spec_class[num_samples_train:num_samples_train + num_samples_valid]
            lst_test_images_class = np.setdiff1d(lst_tmp_spec_class, lst_train_images_class + lst_valid_images_class)

            # Get masks file names for these images
            lst_train_masks_class = get_mask_file_names(lst_train_images_class)
            lst_valid_masks_class = get_mask_file_names(lst_valid_images_class)
            lst_test_masks_class = get_mask_file_names(lst_test_images_class)

            # Get full file paths and append to a list
            lst_train_images_class = [os.path.join(images_dir, item) for item in lst_train_images_class]
            lst_valid_images_class = [os.path.join(images_dir, item) for item in lst_valid_images_class]
            lst_test_images_class = [os.path.join(images_dir, item) for item in lst_test_images_class]

            # Get full file paths of masks
            lst_train_masks_class = [os.path.join(masks_dir, item) for item in lst_train_masks_class]
            lst_valid_masks_class = [os.path.join(masks_dir, item) for item in lst_valid_masks_class]
            lst_test_masks_class = [os.path.join(masks_dir, item) for item in lst_test_masks_class]

            # Append it to a master list
            lst_train_images.extend(lst_train_images_class)
            lst_valid_images.extend(lst_valid_images_class)
            lst_test_images.extend(lst_test_images_class)

            lst_train_masks.extend(lst_train_masks_class)
            lst_valid_masks.extend(lst_valid_masks_class)
            lst_test_masks.extend(lst_test_masks_class)

        # define transformations
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Creating Datasets
        train_dataset = ProductSegmentationDataset(imagePaths=lst_train_images, maskPaths=lst_train_masks,
                                                   transform=transform)
        val_dataset = ProductSegmentationDataset(imagePaths=lst_valid_images, maskPaths=lst_valid_masks,
                                                 transform=transform)
        test_dataset = ProductSegmentationDataset(imagePaths=lst_test_images, maskPaths=lst_test_masks,
                                                  transform=transform)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

