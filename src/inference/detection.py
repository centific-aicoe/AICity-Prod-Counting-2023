import os
import sys
import yolov5

# Adding working directory to path to run the files as standalone
curr_wd = os.getcwd()
if curr_wd not in sys.path:
    sys.path.append(curr_wd)  # add working directory to PATH

from src.inference.sort import Sort


class ProdDetectionandTracking:
    def __init__(self, model_path, tracker_model="SORT"):

        # Load detection model
        self.prod_detect_model = yolov5.load(model_path)

        # Load tracker
        if tracker_model == "SORT":
            self.tracker = Sort(max_age=100, min_hits=3, iou_threshold=0.3)

    def get_prod_detections_df(self, frame, conf_thresh=0.75):
        results = self.prod_detect_model(frame)
        detect_df1 = results.pandas().xyxy[0]

        # update class
        detect_df1["class"] = detect_df1["class"].map(lambda x: x + 1)

        # Filter detections
        detect_df2 = detect_df1[detect_df1["confidence"] > conf_thresh]

        return detect_df2


