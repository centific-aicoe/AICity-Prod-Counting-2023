# AICity-Prod-Counting-2023
This repo contains the code which is part of submission by team Centific for Track-4 AI City Challenge, 2023.

## Environment creation
In order to create an environment to execute all the codes, run the commands below

#### Create a virtualenv environment
The below command will create a folder named **venv** in the current directory which will have the environment

```bash
python -m venv ./venv
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

## Training

## Inference
In order to replicate the leaderboard performance for test set A, execute the code as shown below

```bash
python src/inference/create_submission.py --test_set_dir <test_set_dir> --model_path ./model_weights/best.pt --output_dir <output_dir>
```

There are three arguments that are needed for this script

1. **test_set_dir** - Path to the directory which contains test A videos as shown below

![img.png](assets/test_dir_structure.png)

**Note**: _The file video_id.txt is also consumed by the code. It is expected to be present in the same directory as videos_

2. **model_path** - Path of the model file to be used.model weights used to generate best leaderboard submission are placed at ./model_weights/best.pt

3. **output_dir** - Path to the directory where output file along with the inference videos would be written
