import cv2
import numpy as np


def get_intersection_area(bb1, bb2):
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    product_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])

    percentage_intersection_area = round((intersection_area / product_area), 2)

    return percentage_intersection_area


def get_static_objects(lst_track_ids, master_df, delta_pixels=10):
    static_track_ids = []

    for tmp_track_id in lst_track_ids:

        # Subset the data
        tmp_df1 = master_df[master_df["tracklet_id"] == tmp_track_id].copy()

        if len(tmp_df1) > 0:
            # Computing centers
            tmp_df1["x_center"] = tmp_df1.apply(lambda row: (row["xmin"] + row["xmax"]) / 2, axis=1)
            tmp_df1["y_center"] = tmp_df1.apply(lambda row: (row["ymin"] + row["ymax"]) / 2, axis=1)

            # Computer list of x and y center delta difference
            tmp_lst_x_center = tmp_df1["x_center"].tolist()
            tmp_lst_y_center = tmp_df1["y_center"].tolist()

            # Delta less than 10 pixels
            tmp_lst_delta_x = [np.abs(item - np.mean(tmp_lst_x_center)) <= delta_pixels for item in tmp_lst_x_center]
            tmp_lst_delta_y = [np.abs(item - np.mean(tmp_lst_y_center)) <= delta_pixels for item in tmp_lst_y_center]

            if sum(tmp_lst_delta_x) == len(tmp_df1) and sum(tmp_lst_delta_y) == len(tmp_df1):
                static_track_ids.append(tmp_track_id)

    return static_track_ids