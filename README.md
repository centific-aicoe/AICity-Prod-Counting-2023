# AICity-Prod-Counting-2023
This repo contains the code which is part of submission by team Centific for Track-4 AI City Challenge, 2023.

There are 4 major sections in this markdown
1. [Environment Creation](#environment-creation)
2. [Data Preparation](#data-preparation)
3. [Model Training](#model-training)
4. [Inference](#inference)

If you want to replicate the leaderboard submission, create an environment as per instructions 
in [Environment Creation](#environment-creation) and follow instructions in [Inference](#inference).

For any other steps, look at the relevant section.

## Environment creation
In order to create an environment to execute all the codes, run the commands below

#### Create a virtualenv environment
The below command will create a folder named **venv** in the current directory which will have the environment

```bash
python3 -m venv venv
```

#### Activate the environment
The command below will activate the created environment in above step
```bash
source venv/bin/activate
```

### Install the packages
The command below will install the packages listed in the **requirements.txt** in the activated environment

```bash
pip install -r requirements.txt
```

## Data Preparation
The process of data preparation contains two major steps
1. Creation of Object Detection Dataset
2. Creation of dataset for Yolov5 training 


### Creation of Object Detection Dataset
As part of the AI City challenge, we received a dataset which contains product images and 
segmentation masks as shown below

![dataset_folder.png](assets/dataset_folder.png)

We have developed a code to use these product images and segmentation masks and create 
an object detection dataset.

For every image in object detection dataset, a background image is required. This is sampled from
sample from 3 backgrounds from this [folder](./backgrounds)

In order to generate the object detection dataset, execute the command below
```bash
python src/data_prep/create_synth_detection_data.py --data_dir <data_dir> --background_path ./backgrounds --out_dir <out_dir> --dontocclude
```
There are 4 arguments that are needed to run the above script. 
Details of arguments are given below
1. **data_dir** - The directory which has the raw data. This directory should two folders named **train** and **segmentation_labels** and file named **label.txt** as shown below
![dataset_folder.png](assets/dataset_folder.png)

2. **background_path** - This directory should contain the background files that need to be used

3. **out_dir** - The path where object detection dataset would be saved

4. **dontocclude** - This argument will ensure to paste products on the background so that they are not fully occluded

After executing the code, you will see the following folders & files  in the **out_dir**

![data_generate_output_structure.png](assets/data_generate_output_structure.png)

### Creation of dataset for Yolov5 training
After the objection detection dataset is created as per the above step, We 
need to prepare the data in a folder structure which is compatible for Yolov5 training.
This section explains the steps for the same.

To prepare the data which is ready for Yolov5 custom training, execute the command below
```bash
python python src/data_prep/create_yolo_data_folders.py --input_dir <input_dir> --output_dir <output_dir> \
                                                        --train_ratio 0.7 --valid_ratio 0.15 --test_ratio 0.15
```
There are 5 arguments that are needed to run the above script. 
Details of arguments are given below
1. **input_dir** - The folder path which contains the object detection data. This will be the 
output of the previous step (Creation of Object Detection Dataset)

![data_generate_output_structure.png](assets/data_generate_output_structure.png)

2. **output_dir** - The folder path where files would be saved as per yolov5 structure
3. **train_ratio** - Percentage of data to be used for training
4. **valid_ratio** - Percentage of data to be used for validation
5. **test_ratio** - Percentage of data to be used for test

After executing the code, you will see the following folders & files in the **output_dir**

![yolo_folder_structure.png](assets/yolo_folder_structure.png)

The folder path **output_dir** in this step will be used for yolov5 custom model training. 

## Model Training

## Inference
In order to replicate the leaderboard performance for test set A, execute the code as shown below

```bash
python src/inference/create_submission.py --test_set_dir <test_set_dir> --model_path ./model_weights/best.pt --output_dir <output_dir>
```

There are three arguments that are needed for this script

1. **test_set_dir** - Path to the directory which contains test A videos as shown below

![test_dir_structure.png](assets/test_dir_structure.png)

**Note**: _The file video_id.txt is also consumed by the code. It is expected to be present in the same directory as videos_
2. **model_path** - Path of the model file to be used.model weights used to generate best leaderboard submission are placed at ./model_weights/best.pt
3. **output_dir** - Path to the directory where output file along with the inference videos would be written
