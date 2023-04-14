import os
import sys
import cv2

# Adding working directory to path to run the files as standalone
import pandas as pd

curr_wd = os.getcwd()
if curr_wd not in sys.path:
    sys.path.append(curr_wd)  # add working directory to PATH

from src.inference.roi import RegionofInterest
from src.inference.detection import ProdDetectionandTracking
from src.inference.helpers import get_intersection_area, get_static_objects


def create_inference_info(videofile_path, model_path, output_dir, req_num_frames=-1):

    # Get base video name
    filename = os.path.basename(videofile_path)
    out_filename = "out" + "_" + filename

    # Load Video Meta Information
    cap = cv2.VideoCapture(videofile_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Get the total numer of frames in the video.
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    print("Video path is ", videofile_path)
    print("FPS is ", fps)
    print("Height is ", frame_height)
    print("Width is ", frame_width)
    print("Total Frames ", frame_count)

    if req_num_frames == - 1:
        req_num_frames = frame_count
        print(f"Processing all frames {req_num_frames}")
    else:
        print(f"Processing only first {req_num_frames} frames")

    # Setting start position
    # start_position = 0
    # cap.set(cv2.CAP_PROP_POS_FRAMES, start_position)

    # Initialize videowriting
    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter(os.path.join(output_dir, out_filename),
                             fourcc, fps, (frame_width, frame_height))

    # Instantiate ROI class
    roi_obj = RegionofInterest(fps=fps, reset_interval_secs=0.5)

    # Instantiate detection class
    prod_detect_obj = ProdDetectionandTracking(model_path=model_path)

    # start position
    i = 0

    # List to store results
    lst_results = []

    # Loop over the video and do processing
    while (cap.isOpened()):

        ret, frame = cap.read()

        if ret == True and i <= req_num_frames:

            # Convert frame from BGR to RGB
            req_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Generate roi on the frame
            roi_obj.generate_roi(frame=req_frame, frame_number=i)

            # Get ROI bbox
            roi_bbox = roi_obj.roi_bbox

            # Draw ROI
            cv2.rectangle(frame, (roi_bbox["xmin"], roi_bbox["ymin"]), (roi_bbox["xmax"], roi_bbox["ymax"]),
                          (255, 255, 255), 2)

            # Get prod detections
            detect_df2 = prod_detect_obj.get_prod_detections_df(frame=req_frame)

            # Convert them to tracker required format
            lst_prod_detections = []
            for index, row in detect_df2.iterrows():
                tmp_dict1 = {}
                tmp_dict1["rect_coords"] = [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]
                tmp_dict1["conf"] = row["confidence"]
                tmp_dict1["class"] = row["class"]
                lst_prod_detections.append(tmp_dict1)

            # Update tracker
            trackers, max_obj, obj_ids = prod_detect_obj.tracker(frame=req_frame, detections=lst_prod_detections)

            # Logic to write results
            for item in trackers:
                prod_xmin = int(item[0])
                prod_ymin = int(item[1])
                prod_xmax = int(item[2])
                prod_ymax = int(item[3])
                tracklet_id = item[4]
                class_id = item[5]

                cv2.rectangle(frame, (prod_xmin, prod_ymin), (prod_xmax, prod_ymax), (0, 0, 255), 2)
                cv2.putText(frame, "track id:" + str(tracklet_id) + " class id: " + str(class_id)
                            + " , " + str(round(row["confidence"], 2)),
                            (prod_xmin + 30, prod_ymin + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # # Compute percentage overlap
                # tmp_intersect_area = get_intersection_area(bb1={"x1": roi_bbox["xmin"], "y1": roi_bbox["ymin"],
                #                                                 "x2": roi_bbox["xmax"], "y2": roi_bbox["ymax"]},
                #                                            bb2={"x1": prod_xmin, "y1": prod_ymin,
                #                                                 "x2": prod_xmax, "y2": prod_ymax})

                # Check if detection is inside ROI
                if (prod_xmin >= roi_bbox["xmin"] and prod_xmin <= roi_bbox["xmax"]) and \
                        (prod_ymin >= roi_bbox["ymin"] and prod_ymin <= roi_bbox["ymax"]) and \
                        (prod_xmax >= roi_bbox["xmin"] and prod_xmax <= roi_bbox["xmax"]) and \
                        (prod_ymax >= roi_bbox["ymin"] and prod_ymax <= roi_bbox["ymax"]):

                    # Save the results
                    tmp_dict3 = {}
                    tmp_dict3["xmin"] = prod_xmin
                    tmp_dict3["ymin"] = prod_ymin
                    tmp_dict3["xmax"] = prod_xmax
                    tmp_dict3["ymax"] = prod_ymax
                    tmp_dict3["tracklet_id"] = tracklet_id
                    tmp_dict3["class_id"] = class_id
                    tmp_dict3["frame_number"] = i

                    lst_results.append(tmp_dict3)

            # Write to video
            output.write(frame)
            i += 1
        else:
            break

    cap.release()
    output.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    # Create a dataframe
    if len(lst_results) > 0:
        prod_results_df = pd.DataFrame(lst_results)

        # Logic to create one instance per product
        prod_results_df_final = prod_results_df.groupby(["tracklet_id", "class_id"],
                                                        as_index=False).agg(min_frame_number=("frame_number", "min"),
                                                                            max_frame_number=("frame_number", "max"),
                                                                            num_frames=("frame_number", "count"))

        # Create mean of the min and max
        # Applying a minimum detection of 10 seconds
        # prod_filt_df1 = prod_results_df_final[prod_results_df_final["num_frames"] > 5].copy()
        prod_filt_df1 = prod_results_df_final.copy()

        # Get static list of static ids
        # Filtering IDs which are present in more than 3 frames
        lst_track_ids = prod_filt_df1[prod_filt_df1["num_frames"] > 10]["tracklet_id"].tolist()
        lst_static_track_ids = get_static_objects(lst_track_ids, prod_results_df, delta_pixels=10)

        # Remove the static track ids
        prod_filt_df2 = prod_filt_df1[~prod_filt_df1["tracklet_id"].isin(lst_static_track_ids)].copy()

        # Get final frame id
        if len(prod_filt_df2) > 0:
            prod_filt_df2["frame_id"] = prod_filt_df2.apply(lambda row :
                                                            int((row["min_frame_number"]+row["max_frame_number"])/2),
                                                            axis=1)
            return prod_filt_df2
        else:
            return pd.DataFrame()

    else:
        return pd.DataFrame()


videofile_path = '/Users/anudeep/Desktop/AI_city_challenge/AIC23_Track4_Automated_Checkout/testA/testA_2.mp4'
model_path = "/Users/anudeep/Desktop/AI_city_challenge/first_cut_model_9thMarch/best.pt"
req_num_frames = -1
output_dir = "/Users/anudeep/Desktop/AI_city_challenge/submissions/best_results_replication"
