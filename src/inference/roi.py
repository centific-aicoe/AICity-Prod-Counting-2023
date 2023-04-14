import os
import sys
from collections import deque
import yolov5
import pandas as pd

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Adding working directory to path to run the files as standalone
curr_wd = os.getcwd()
if curr_wd not in sys.path:
    sys.path.append(curr_wd)  # add working directory to PATH

from src.inference.helpers import get_intersection_area


class RegionofInterest:
    def __init__(self, fps, upper_horizontal_line_coord=10000, lower_horizontal_line_coord=0,
                 left_vertical_line_coord=0, right_vertical_line_coord=0,
                 reset_interval_secs=5):

        # Initialize co-ordinates
        self.upper_horizontal_line_coord = upper_horizontal_line_coord
        self.lower_horizontal_line_coord = lower_horizontal_line_coord
        self.left_vertical_line_coord = left_vertical_line_coord
        self.right_vertical_line_coord = right_vertical_line_coord

        # Load yolov5 model
        self.model = yolov5.load('yolov5m.pt')

        # Parameters for ROI reset
        self.reset_interval_secs = reset_interval_secs  # 2 minutes- contains 2 * 60
        self.reset_interval_frames = int(reset_interval_secs * fps)
        self.buffer_deque_results = deque(maxlen=self.reset_interval_frames)
        self.roi_bbox = {"xmin": int(left_vertical_line_coord), "ymin": 0,
                         "xmax": int(right_vertical_line_coord), "ymax": int(lower_horizontal_line_coord)}

        print(f"Reset interval frames is {self.reset_interval_frames}")

    def get_yolov5_detections(self, frame):
        results = self.model(frame)
        return results

    def generate_roi(self, frame, frame_number):

        # Get frame details
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        # Get yolov5 pretrained detections
        results = self.get_yolov5_detections(frame)

        # Convert to dataframe
        df1 = results.pandas().xyxy[0]

        # Compute lines
        # Get all the list of persons with threshold > 0.8
        req_person_df = df1[(df1["name"] == "person") & (df1["confidence"] >= 0.8)].copy()
        non_person_df = df1[(df1["name"] != "person") & (df1["confidence"] > 0.25)].copy()

        # Get hands information
        with mp_hands.Hands(model_complexity=1,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5) as hands:
            hand_results = hands.process(frame)

        # Check overlap with list of non-persons with a threshold of 0.25
        for index, row in req_person_df.iterrows():
            xmin1 = row["xmin"]
            ymin1 = row["ymin"]
            xmax1 = row["xmax"]
            ymax1 = row["ymax"]

            if len(non_person_df) > 0:
                # Compute distances
                non_person_df["perc_intersection_area"] = non_person_df.apply(
                    lambda row: get_intersection_area(bb1={"x1": xmin1, "y1": ymin1,
                                                           "x2": xmax1, "y2": ymax1},
                                                      bb2={"x1": row["xmin"], "y1": row["ymin"],
                                                           "x2": row["xmax"], "y2": row["ymax"]}), axis=1)

                # Eliminated items with no-overlap
                filt_df1 = non_person_df[(non_person_df["perc_intersection_area"] > 0.85)
                                         & (non_person_df["perc_intersection_area"] < 1)].copy()

                # Get minimum of ymin
                if len(filt_df1) > 0:
                    # tmp_upper_horizontal_line_coord = filt_df1["ymin"].min()
                    # tmp_lower_horizontal_line_coord = filt_df1["ymax"].max()
                    #
                    # if tmp_upper_horizontal_line_coord > 0:
                    #     self.upper_horizontal_line_coord = min(tmp_upper_horizontal_line_coord,
                    #                                            self.upper_horizontal_line_coord)
                    #
                    # if tmp_lower_horizontal_line_coord < frame_height:
                    #     self.lower_horizontal_line_coord = max(tmp_lower_horizontal_line_coord,
                    #                                            self.lower_horizontal_line_coord)

                    if hand_results.multi_handedness is not None:
                        if len(hand_results.multi_handedness) == 2:
                            tmp_dict2 = row.to_dict()
                            tmp_dict2["tag"] = "object overlap"
                            tmp_dict2["frame_number"] = frame_number
                            self.buffer_deque_results.append(tmp_dict2)

                else:
                    if hand_results.multi_handedness is not None:
                        if len(hand_results.multi_handedness) == 2:
                            tmp_dict1 = row.to_dict()
                            tmp_dict1["tag"] = "no object overlap"
                            tmp_dict1["frame_number"] = frame_number
                            self.buffer_deque_results.append(tmp_dict1)

        # Concatentate results
        results_df = pd.DataFrame(list(self.buffer_deque_results))

        # Computing vertical lines
        if len(results_df) > 0:

            # Compute mid point
            results_df["person_center_x"] = results_df.apply(lambda row: (row["xmax"] + row["xmin"]) / 2, axis=1)
            results_df["person_center_y"] = results_df.apply(lambda row: (row["ymax"] + row["ymin"]) / 2, axis=1)

            # Compute width
            results_df["person_box_width"] = results_df.apply(lambda row: (row["xmax"] - row["xmin"]), axis=1)
            results_df["person_box_height"] = results_df.apply(lambda row: (row["ymax"] - row["ymin"]), axis=1)

            # Take mean of no product overlap and product overlap
            overlap_df = results_df[results_df["tag"] == "object overlap"]
            no_overlap_df = results_df[results_df["tag"] == "no object overlap"]

            # Compute means separately
            #             tmp_means_info = compute_hybrid_mean(no_overlap_df, overlap_df)
            #             x_person_center = tmp_means_info["person_center_x"]
            #             y_person_center = tmp_means_info["person_center_y"]
            #             person_box_width = tmp_means_info["person_box_width"]
            #             person_box_height = tmp_means_info["person_box_height"]

            # Compute mean center # straight mean
            x_person_center = results_df["person_center_x"].mean()
            y_person_center = results_df["person_center_y"].mean()
            person_box_width = results_df["person_box_width"].mean()
            person_box_height = results_df["person_box_height"].mean()

            left_vertical_line_coord = x_person_center - (person_box_width / 2)
            right_vertical_line_coord = x_person_center + (person_box_width / 2)

            # Logic for horizontal lines
            lower_horizontal_line_coord = y_person_center

            self.roi_bbox = {"xmin": int(left_vertical_line_coord), "ymin": 0,
                             "xmax": int(right_vertical_line_coord), "ymax": int(lower_horizontal_line_coord)}