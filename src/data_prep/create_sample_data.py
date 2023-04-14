import os
import argparse
import shutil


def prep_sample_data(data_dir, output_dir):

    # train directory name and masks directory name
    train_dir_name = "train"
    masks_dir_name = "segmentation_labels"

    # Get the folder details of images and masks
    images_dir = os.path.join(data_dir, train_dir_name)
    masks_dir = os.path.join(data_dir, masks_dir_name)

    # Select n= 20 images and its masks per class
    num_per_class = 15

    # Empty list to store all images information
    lst_req_images = []

    for i in range(1, 117):

        # convert to string
        tmp_str1 = str(i)

        # convert to the required format
        tmp_str2 = tmp_str1.zfill(5)

        # get all the files for a specific class
        lst_tmp_spec_class = [item for item in os.listdir(images_dir) if item.startswith(tmp_str2)]

        # select n objects
        lst_req_tmp_spec_class = lst_tmp_spec_class[:num_per_class]

        # Add this to a master list
        lst_req_images.extend(lst_req_tmp_spec_class)

    # save the images and masks to a output folder
    # Create directories if not exists
    os.makedirs(os.path.join(output_dir, train_dir_name), exist_ok=True)
    os.makedirs(os.path.join(output_dir, masks_dir_name), exist_ok=True)

    for img_filename in lst_req_images:

        # Get image
        class_name, rem_part = img_filename.split("_")

        # separate the extension
        tmp_id, file_ext = rem_part.split(".")

        # segmentation file name
        seg_filename = class_name + "_" + tmp_id + "_seg"+ "." + file_ext

        # copy the files to a new directory
        src_file_path1 = os.path.join(images_dir, img_filename)
        src_file_path2 = os.path.join(masks_dir, seg_filename)

        # destination paths
        dest_file_path1 = os.path.join(output_dir, train_dir_name, img_filename)
        dest_file_path2 = os.path.join(output_dir, masks_dir_name, seg_filename)

        shutil.copy(src_file_path1, dest_file_path1)
        shutil.copy(src_file_path2, dest_file_path2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',
                        help='Input Directory for raw data',
                        required=True,
                        type=str)

    parser.add_argument('--output_dir',
                        help='Directory to save sample data',
                        required=True,
                        type=str)

    # Reading the arguments
    args = parser.parse_args()

    # Reading the arguments
    data_dir = args.data_dir
    output_dir = args.output_dir

    # function call
    prep_sample_data(data_dir, output_dir)



