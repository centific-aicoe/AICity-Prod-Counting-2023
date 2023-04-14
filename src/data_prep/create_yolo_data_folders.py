import os
import random
import sys
import glob
import shutil
import yaml
import argparse

# Adding working directory to path to run the files as standalone
curr_wd = os.getcwd()
if curr_wd not in sys.path:
    sys.path.append(curr_wd)  # add working directory to PATH


def copy_to_designated_folders(lst_req_items, output_dir, tag):
    for img_file, ann_file in lst_req_items:
        # Copying image
        shutil.copy(src=img_file, dst=os.path.join(output_dir, tag, "images"))

        # Copying label
        shutil.copy(src=ann_file, dst=os.path.join(output_dir, tag, "labels"))


def create_yolo_format_data_folders(input_dir, output_dir, train_ratio, valid_ratio, test_ratio):

    # Input directory should contain two folders and labels.txt file
    images_dir = os.path.join(input_dir, "images")
    anno_dir = os.path.join(input_dir, "labels")
    labels_file_path = os.path.join(input_dir, "labels.txt")

    if not (os.path.exists(images_dir) and os.path.exists(anno_dir) and os.path.exists(labels_file_path)):
        raise Exception("One or many of sub folders images, labels and labels.txt is missing in input_dir")

    print(f"Using train, valid and test ratios as {train_ratio}, {valid_ratio}, {test_ratio} respectively")

    # Get list of images
    img_files = glob.glob(os.path.join(images_dir, '*.jpg'))
    anno_files = [os.path.join(anno_dir, os.path.basename(item).replace(".jpg", ".txt")) for item in img_files]

    # Combine both into one list
    lst_img_ann = list(zip(img_files, anno_files))

    # Shuffle images
    random.Random(4).shuffle(lst_img_ann)

    # Check the sanity
    if sum([train_ratio, valid_ratio, test_ratio]) > 1:
        raise Exception("Sum of train, valid and test ratio should be equal to 1")

    # number of data points
    n_train = int(train_ratio * len(lst_img_ann))
    n_valid = int(valid_ratio * len(lst_img_ann))
    n_test = len(lst_img_ann) - n_train - n_valid

    # Subset the list into three lists
    lst_train = lst_img_ann[:n_train]
    lst_valid = lst_img_ann[n_train:n_train + n_valid]
    lst_test = lst_img_ann[n_train+n_valid:n_train+n_valid+n_test]

    # Create three folders train, val and test under the output directory
    req_folders = ["train", "val", "test"]
    for folder in req_folders:
        tmp_dir = os.path.join(output_dir, folder)
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

        # Creating a fresh directory
        os.makedirs(tmp_dir, exist_ok=True)
        os.makedirs(os.path.join(tmp_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(tmp_dir, "labels"), exist_ok=True)

    # Copying all the images from input directory to relevant train, val and test directories
    copy_to_designated_folders(lst_train, output_dir, tag="train")
    copy_to_designated_folders(lst_valid, output_dir, tag="val")
    copy_to_designated_folders(lst_test, output_dir, tag="test")

    print(f"Created yolo format data at location: {output_dir}")

    # Create the yaml required for yolo
    tmp_dict1 = {}
    for item in req_folders:
        # yolov5 directory will be present in the root directory
        tmp_dict1[item] = os.path.relpath(os.path.join(output_dir, item), "yolov5")

    # Get the names of classes
    with open(labels_file_path, "r") as f:
        labels_info = f.readlines()

    # Create a dictionary and append all the info
    tmp_dict2 = {}
    for item in labels_info:
        class_label, class_name = item.rstrip("\n").split("~")
        tmp_dict2[int(class_label)] = class_name

    # Combining it to the main dictionary
    tmp_dict1["names"] = tmp_dict2
    with open(os.path.join(output_dir, 'prod_detection.yaml'), 'w') as file:
        yaml.dump(tmp_dict1, file)


def parse_args():
    parser = argparse.ArgumentParser(description="Create yolo fold structure and yaml")
    parser.add_argument('--input_dir',
                        type=str,
                        help='Folder which contains created synthetic data. '
                             'Two sub folders images and labels and a file labels.txt are expected',
                        required=True)
    parser.add_argument('--output_dir',
                        type=str,
                        help='Folder where yolo format folders and files can be created',
                        required=True)
    parser.add_argument('--train_ratio',
                        type=float,
                        help='Ratio of training examples to the total data',
                        default=0.7)
    parser.add_argument('--valid_ratio',
                        type=float,
                        help='Ratio of valid examples to the total data',
                        default=0.15)
    parser.add_argument('--test_ratio',
                        type=float,
                        help='Ratio of test examples to the total data',
                        default=0.15)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Reading the arguments
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    test_ratio = args.test_ratio

    create_yolo_format_data_folders(input_dir, output_dir, train_ratio, valid_ratio, test_ratio)
