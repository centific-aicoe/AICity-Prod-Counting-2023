

def get_mask_file_names(lst_image_filenames):

    # empty list to store file names
    lst_mask_filenames = []

    for img_filename in lst_image_filenames:

        # Get image
        class_name, rem_part = img_filename.split("_")

        # separate the extension
        tmp_id, file_ext = rem_part.split(".")

        # segmentation file name
        seg_filename = class_name + "_" + tmp_id + "_seg" + "." + file_ext

        # Append it to list
        lst_mask_filenames.append(seg_filename)

    return lst_mask_filenames


def fancy_print(x, n=75):
    print("-" * n)
    print(x)
    print("-" * n)
