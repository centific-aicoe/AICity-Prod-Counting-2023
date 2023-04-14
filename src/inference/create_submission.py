import os
import sys
import glob
import time

curr_wd = os.getcwd()
if curr_wd not in sys.path:
    sys.path.append(curr_wd)  # add working directory to PATH

from src.inference.generate_inference import create_inference_info


def prepare_submission(test_set_dir, model_path, output_dir):

    # Read files from test set dir
    # Videos and a file video_id.txt is expected
    # Get list of videos
    video_files = glob.glob(os.path.join(test_set_dir, "*.mp4"))
    video_files.sort()

    # Get video id info
    with open(os.path.join(test_set_dir, "video_id.txt"), "r") as f:
        lst_video_ids = f.readlines()

    # iterate over list
    video_info_dict = {}
    for item in lst_video_ids:
        video_id, video_filename = item.rstrip("\n").split(" ")
        video_info_dict[video_filename] = video_id

    # Loop over videos and get information
    for tmp_video_path in video_files:
        # Start timer
        start = time.time()

        # Process video and get results
        tmp_prod_results_df = create_inference_info(tmp_video_path, model_path, output_dir, req_num_frames=-1)

        # Get the video id
        req_base_filename = os.path.basename(tmp_video_path)
        req_video_id = video_info_dict[req_base_filename]

        # Temp string to save results
        results_str = ""

        # Loop over dataframe and write to text file
        for index, row in tmp_prod_results_df.iterrows():
            results_str = results_str + " ".join([req_video_id, str(int(row["class_id"])),
                                str(int(row["frame_id"]))]) +"\n"

        # Write to a text file
        if len(results_str) > 0:
            with open(os.path.join(output_dir, "submission.txt"), "a") as f:
                f.write(results_str)

        end = time.time()
        print(f"Time taken for processing {req_base_filename} is {int((end-start)/60)} minutes")


if __name__ == '__main__':
    test_set_dir = "/Users/anudeep/Desktop/AI_city_challenge/AIC23_Track4_Automated_Checkout/testA"
    model_path = "/Users/anudeep/Desktop/AI_city_challenge/first_cut_model_9thMarch/best.pt"
    output_dir = "/Users/anudeep/Desktop/AI_city_challenge/submissions/results_replication_full"



