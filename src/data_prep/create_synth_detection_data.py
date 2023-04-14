# Lot of code adapted from link below
# https://github.com/debidatta/syndata-generation/blob/aeec2102566f8ffcc7309b46e1ba1c381c23d9ba/dataset_generator.py

import os
import glob
import sys
import random
import argparse

import pandas as pd
from PIL import Image
import cv2
import numpy as np
import signal
import shutil

from collections import namedtuple
from multiprocessing import Pool
from functools import partial

# Adding working directory to path to run the files as standalone
curr_wd = os.getcwd()
if curr_wd not in sys.path:
    sys.path.append(curr_wd)  # add working directory to PATH

from src.data_prep.synthetic_config import *

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')


def PIL2array1C(img):
    '''Converts a PIL image to NumPy Array
    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    '''
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0])


def PIL2array3C(img):
    '''Converts a PIL image to NumPy Array
    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    '''
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 3)


def overlap(a, b):
    '''Find if two bounding boxes are overlapping or not. This is determined by maximum allowed
       IOU between bounding boxes. If IOU is less than the max allowed IOU then bounding boxes
       don't overlap
    Args:
        a(Rectangle): Bounding box 1
        b(Rectangle): Bounding box 2
    Returns:
        bool: True if boxes overlap else False
    '''
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)

    if (dx >= 0) and (dy >= 0) and float(dx * dy) > MAX_ALLOWED_IOU * (a.xmax - a.xmin) * (a.ymax - a.ymin):
        return True
    else:
        return False


def get_list_of_images_and_masks(data_dir):

    # data directory should contain two folders
    # 1. train - this contains images
    # 2. segmentation_labels - this contains masks

    # Get list of images
    img_files = glob.glob(os.path.join(data_dir, "train", '*.jpg'))

    # Random shuffle
    random.Random(4).shuffle(img_files)

    # get mask files details as well
    mask_files = [os.path.join(data_dir, "segmentation_labels",
                               os.path.basename(item).replace(".jpg", "_seg.jpg")) for item in img_files]

    return img_files, mask_files


def get_labels(img_files):

    # Empty list to get labels info
    labels = []

    for img_file in img_files:
        label = os.path.basename(img_file).split("_")[0]

        # Convert to class number for yolo
        # yolov5 starts labelling from 0
        class_number = int(label.lstrip("0")) - 1
        labels.append(class_number)

    return labels


def write_labels_file(out_dir, labels, labels_info_dict):
    '''Writes the labels file which has the name of an object on each line
    Args:
        out_dir(string): Experiment directory where all the generated images, annotation and imageset
                         files will be stored
        labels(list): List of labels. This will be useful while training an object detector
    '''
    unique_labels = sorted(set(labels))

    with open(os.path.join(out_dir, 'labels.txt'), 'w') as f:
        for i, label in enumerate(unique_labels):
            f.write('%s~%s\n' % (label, labels_info_dict[label]))


def compute_normalized_coordinates(xmin, xmax, ymin, ymax, width, height):

    # Adjusting boxes caused due to generating x,y outside the frame
    xmin = max(xmin, 0) # Co-ordinates should start at zero; to address negative xmin for truncation
    ymin = max(ymin, 0) # Co-ordinates should start at zero;
    xmax = min(width, xmax)
    ymax = min(height, ymax)

    # Computing the center of the bounding box
    x_center = (xmin + xmax)/2
    y_center = (ymin + ymax)/2

    # Compute width and height of the box
    width_bbox = xmax - xmin
    height_bbox = ymax - ymin

    # Normalize the co-ordinates
    x_center_norm = x_center/width
    y_center_norm = y_center/height

    # Normalize the width # only half width is needed
    width_bbox_norm = width_bbox/(width)
    height_bbox_norm = height_bbox/(height)

    return x_center_norm, y_center_norm, width_bbox_norm, height_bbox_norm


def init_worker():
    '''
    Catch Ctrl+C signal to termiante workers
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def create_image_anno(objects, out_img_file, out_anno_file, bg_file, w, h, rotation_augment, blending_list, dontocclude):

    # print("Working on image file: ", out_img_file)
    # w = 1920
    # h = 1080

    while True:
        # Read the background file
        background = Image.open(bg_file)
        background = background.resize((w, h))

        backgrounds = []
        # Loop over blending list
        for i in range(len(blending_list)):
            backgrounds.append(background.copy())

        # keep a control on occlusion
        if dontocclude:
            already_syn = []

        assert len(objects) > 0

        # yolov5 annotation string
        annotation_info = ""

        # Loop over all objects
        for index, item in enumerate(objects):

            # Read product and its mask
            tmp_prod = Image.open(item[0])
            tmp_mask = Image.open(item[1])

            # Get object details
            o_w, o_h = tmp_prod.size

            # Generate random location on the image
            # Validate occlusion
            attempt = 0
            while True:
                attempt += 1
                x = random.randint(int(-MAX_TRUNCATION_FRACTION * o_w), int(w - o_w + MAX_TRUNCATION_FRACTION * o_w))
                y = random.randint(int(-MAX_TRUNCATION_FRACTION * o_h), int(h - o_h + MAX_TRUNCATION_FRACTION * o_h))

                # for control on occlusion
                if dontocclude:

                    # Variable intialized as no occlusion found
                    found = True

                    # loop over existing syntheized boxes
                    for prev in already_syn:
                        ra = Rectangle(prev[0], prev[2], prev[1], prev[3])
                        rb = Rectangle(x, y, x + o_w, y + o_h)

                        # Check for overlap more than threshold
                        if overlap(ra, rb): # If overlap found then break out of for loop and try generating again
                            found = False
                            break

                    if found: # despite looping nothing overlapping found, break from while loop and save co-ordinates
                        break
                else:
                    break

                if attempt == MAX_ATTEMPTS_TO_SYNTHESIZE:
                    break

            if dontocclude:
                already_syn.append([x, x+o_w, y, y+o_h])

            # Create annotation format
            class_label = item[2]

            # Computing normalized co-ordinates
            x_center_norm, y_center_norm, width_bbox_norm, height_bbox_norm = \
                compute_normalized_coordinates(xmin=x, xmax=x+o_w, ymin=y, ymax=y+o_h, width=w, height=h)

            # Combine the details to string
            lst_req_values = [class_label, x_center_norm, y_center_norm, width_bbox_norm, height_bbox_norm]
            lst_req_str = " ".join([str(x) for x in lst_req_values])+"\n"

            # Append to annotation
            annotation_info = annotation_info + lst_req_str

            # Apply blending
            for i in range(len(blending_list)):
                if blending_list[i] == 'simplepaste' or blending_list[i] == 'motion':
                    backgrounds[i].paste(tmp_prod, (x, y), tmp_mask)
                # elif blending_list[i] == 'poisson':
                #     offset = (y, x)
                #     img_mask = PIL2array1C(tmp_mask)
                #     img_src = PIL2array3C(tmp_prod).astype(np.float64)
                #     img_target = PIL2array3C(backgrounds[i])
                #     img_mask, img_src, offset_adj \
                #         = create_mask(img_mask.astype(np.float64),
                #                       img_target, img_src, offset=offset)
                #     background_array = poisson_blend(img_mask, img_src, img_target,
                #                                      method='normal', offset_adj=offset_adj)
                #     backgrounds[i] = Image.fromarray(background_array, 'RGB')
                elif blending_list[i] == 'gaussian':
                    backgrounds[i].paste(tmp_prod, (x, y),
                                         Image.fromarray(cv2.GaussianBlur(PIL2array1C(tmp_mask), (5, 5), 2)))
                elif blending_list[i] == 'box':
                    backgrounds[i].paste(tmp_prod, (x, y), Image.fromarray(cv2.blur(PIL2array1C(tmp_mask), (3, 3))))

        if attempt == MAX_ATTEMPTS_TO_SYNTHESIZE:
            print("Inside max attempts")
            continue
            print("hello")
        else:
            break

    # Write images and annotations to folder
    for i in range(len(blending_list)):

        # Writing image to file
        backgrounds[i].save(out_img_file.replace('simplepaste', blending_list[i]))

        # Writing annotation to file
        with open(out_anno_file.replace('simplepaste', blending_list[i]), "w") as f:
            f.write(annotation_info)


def generate_syn_data(img_files, mask_files, labels, background_path, img_dir, anno_dir,
                      rotation_augment, dontocclude):

    # Get details of available background files
    background_files = glob.glob(os.path.join(background_path, "*.png"))

    print("Number of background images :", len(background_files))

    # Create a list of all available files
    lst_img_mask_label = list(zip(img_files, mask_files, labels))
    random.Random(4).shuffle(lst_img_mask_label)

    # Initialization to track
    idx = 0
    out_img_files = []
    out_anno_files = []
    lst_params = []

    # Loop over all items till items are exhausted
    while len(lst_img_mask_label) > 0:

        # initialize empty objects list
        objects = []

        # Get number of objects
        n = min(random.randint(MIN_NO_OF_OBJECTS, MAX_NO_OF_OBJECTS), len(lst_img_mask_label))

        # Add list of objects
        for i in range(n):
            objects.append(lst_img_mask_label.pop())

        idx += 1

        # Select a random background
        bg_file = random.choice(background_files)

        for blur in BLENDING_LIST:
            out_img_file = os.path.join(img_dir, 'image_%i_%s.jpg' % (idx, blur))
            out_anno_file = os.path.join(anno_dir, 'image_%i_%s.txt' % (idx, blur))
            params = (objects, out_img_file, out_anno_file, bg_file)

            lst_params.append(params)
            out_img_files.append(out_img_file)
            out_anno_files.append(out_anno_file)

    # Printing total number of images to be created
    print("Total numbers of images to be created:", len(lst_params))

    # Create partial function for parallel execution
    partial_func = partial(create_image_anno, w=WIDTH, h=HEIGHT, rotation_augment=rotation_augment,
                           blending_list=BLENDING_LIST, dontocclude=dontocclude)

    # Set the number of workers
    num_workers = int(os.cpu_count()/2)
    print(f"Number of workers being used are {num_workers}")

    # initialize pool
    p = Pool(num_workers, init_worker)
    try:
        p.starmap(partial_func, lst_params)
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        p.terminate()
    else:
        p.close()
    p.join()

    return out_img_files, out_anno_files


def create_synthetic_dataset(data_dir, background_path, out_dir, rotation_augment, dontocclude):

    # Get list of image files
    img_files, mask_files = get_list_of_images_and_masks(data_dir)

    # Read master labels from text file
    # data_dir should contain a label.txt file
    # Check if file exists
    labels_file_path = os.path.join(data_dir, "label.txt")

    if not os.path.exists(labels_file_path):
        raise FileNotFoundError(labels_file_path)

    # Creating output directory
    anno_dir = os.path.join(out_dir, 'labels')
    img_dir = os.path.join(out_dir, 'images')

    # Folders to be cleaned up if exists
    cleanup_folders = [anno_dir, img_dir]
    for tmp_directory in cleanup_folders:
        if os.path.exists(tmp_directory):
            print("{} folder already exists. Removing it to create freshly synthesized data"
                  .format(tmp_directory))
            shutil.rmtree(tmp_directory)

    if not os.path.exists(os.path.join(anno_dir)):
        os.makedirs(anno_dir, exist_ok=True)
    if not os.path.exists(os.path.join(img_dir)):
        os.makedirs(img_dir, exist_ok=True)

    # Reading all labels and writing only those relevant to our imageset
    with open(labels_file_path, "r") as f:
        labels_info = f.readlines()

    # Get it into a dictionary form
    labels_info_dict = {int(item.rstrip("\n").split(",")[1])-1:item.rstrip("\n").split(",")[0] for item in labels_info}

    # Get list of labels
    labels = get_labels(img_files)

    # Write labels file
    write_labels_file(out_dir, labels, labels_info_dict)

    # Call function to generate data
    out_img_files, out_anno_files = generate_syn_data(img_files, mask_files, labels, background_path,
                                                      img_dir, anno_dir, rotation_augment, dontocclude)

    # Write information to files
    tmp_df1 = pd.DataFrame({"image_files": out_img_files, "annotation_files": out_anno_files})
    tmp_df1.to_csv(os.path.join(out_dir, "generated_files_info.csv"), index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Create synthetic object detection dataset")
    parser.add_argument('--data_dir',
                        type=str,
                        help='Folder which contains two sub folders train and segmentation_labels',
                        required=True)
    parser.add_argument('--background_path',
                        type=str,
                        help='Folder which background images as .png files',
                        required=True)
    parser.add_argument('--out_dir',
                        type=str,
                        help='Folder where generated synthetic data would be saved',
                        required=True)
    parser.add_argument('--rotation_augment',
                        help="Add rotation augmentation.Default is to add rotation augmentation.",
                        action="store_true")
    parser.add_argument('--dontocclude',
                        help="Add objects without occlusion. Default is to produce occlusions",
                        action="store_true")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Reading the arguments
    args = parse_args()
    data_dir = args.data_dir
    background_path = args.background_path
    out_dir = args.out_dir
    rotation_augment = args.rotation_augment
    dontocclude = args.dontocclude

    # background_path = "datasets/backgrounds"
    # data_dir = "datasets"
    # out_dir = "datasets/synthetic_data"

    create_synthetic_dataset(data_dir, background_path, out_dir, rotation_augment, dontocclude)