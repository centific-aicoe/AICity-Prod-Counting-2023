import os
import sys
import argparse
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.ai.ml import command
from azure.ai.ml import Input, Output, dsl
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
import webbrowser

from dotenv import load_dotenv
load_dotenv()

# Adding working directory to path to run the files as standalone
curr_wd = os.getcwd()
if curr_wd not in sys.path:
    sys.path.append(curr_wd)  # add working directory to PATH


def run_train_pipeline(cloud):

    if cloud:
        # ********************** Connection to ML Workspace ******************** #
        # The DefaultAzureCredential class looks for the following environment variables
        # and uses the values when authenticating as the service principal:
        # 1.AZURE_CLIENT_ID - The client ID returned when you created the service principal.
        # 2.AZURE_TENANT_ID - The tenant ID returned when you created the service principal.
        # 3.AZURE_CLIENT_SECRET - The password/credential generated for the service principal.
        ml_client = MLClient(DefaultAzureCredential(),
                             os.environ.get("SUBSCRIPTION_ID"),
                             os.environ.get("RESOURCE_GROUP"),
                             os.environ.get("WORKSPACE_NAME"))

        # *********** Create environment for the pipeline ******** #
        custom_env_name = "aicity_challenge_detection"
        pipeline_job_env = Environment(
            name=custom_env_name,
            description="Custom environment for AI City challenge",
            conda_file="environment.yml",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.0.3-cudnn8-ubuntu18.04",
        )
        ml_client.environments.create_or_update(pipeline_job_env)

        # Using existing environment
        #pipeline_job_env = ml_client.environments._get_latest_version(name=custom_env_name)

        # *********** Data Path **************** #
        # === Note on path ===
        # can be can be a local path or a cloud path.
        # AzureML supports https://`, `abfss://`, `wasbs://` and `azureml://` URIs.
        # Local paths are automatically uploaded to the default datastore in the cloud.
        # More details on supported paths: https://docs.microsoft.com/azure/machine-learning/how-to-read-write-data-v2#supported-paths

        # Possible Paths for Data:
        # Blob: https://<account_name>.blob.core.windows.net/<container_name>/<folder>/<file> -- Not working in SDK
        # Datastore: azureml://datastores/<data_store_name>/paths/<path>' -- Tested and working
        # Data Asset: azureml:<my_data>:<version>  --- Tested and Not Working
        # Can be a local path on the machine as well --- Tested and Working

        # Recommended way for cloud training would be to use datastore
        # ******** Configure the Job ********** #
        data_process_job = command(
            name="yolov5_dataprep",
            inputs={
                "input_dir": Input(type=AssetTypes.URI_FOLDER,
                                   path="azureml://datastores/aicity_synthetic_data/paths/gaussian_synthetic_data"),
                # "input_dir": Input(type=AssetTypes.URI_FOLDER,
                #                    path="./datasets/synthetic_data"),
                "train_ratio": 0.7,
                "valid_ratio": 0.15,
                "test_ratio": 0.15
            },
            outputs={
                "output_dir": Output(type=AssetTypes.URI_FOLDER)
            },
            code=".",
            command="python src/data_prep/create_yolo_data_folders.py --input_dir ${{inputs.input_dir}} \
                                --output_dir ${{outputs.output_dir}} --train_ratio ${{inputs.train_ratio}} \
                                --valid_ratio ${{inputs.valid_ratio}} --test_ratio ${{inputs.test_ratio}}",
            environment=pipeline_job_env,
            compute="Image-Classifier",
        )

        # Yolo training job
        training_job = command(
            name="yolov5_model_training",
            inputs={
                "input_dir": Input(type=AssetTypes.URI_FOLDER, mode="download"), # yaml will get edited to change paths
                                                                                 # write is only supported with download
                "img": 640,
                "batch_size": 16,
                "weights": "yolov5m.pt",
                "epochs": 70,
                "device": 0,
                "name": "training"
            },
            outputs={
                "output_dir": Output(type=AssetTypes.URI_FOLDER)
            },
            code="yolov5",
            command="python train.py --img ${{inputs.img}}  --batch-size ${{inputs.batch_size}} \
                                    --weights ${{inputs.weights}} --epochs ${{inputs.epochs}} \
                                    --device ${{inputs.device}} \
                                    --project ${{outputs.output_dir}} --name ${{inputs.name}} \
                                    --input_dir ${{inputs.input_dir}}",
                                    # add the folder that needs to be mounted as input argument
                                    # add this as extra argument to train.py (make change accordingly in train.py)
            environment=pipeline_job_env,
            compute="Image-Classifier",
        )

        # Measure Test Performance
        test_job = command(
            name="test_performance",
            inputs={
                "input_dir": Input(type=AssetTypes.URI_FOLDER, mode="download"),
                "model_dir": Input(type=AssetTypes.URI_FOLDER),  # yaml will get edited to change paths
                # write is only supported with download
                "img": 640,
                "batch_size": 16,
                "device": 0,
                "task": "test",
                "name": "test_performance"
            },
            outputs={
                "output_dir": Output(type=AssetTypes.URI_FOLDER)
            },
            code="yolov5",
            command="python val.py --img ${{inputs.img}}  --batch-size ${{inputs.batch_size}} \
                                            --device ${{inputs.device}} --task ${{inputs.task}} \
                                            --project ${{outputs.output_dir}} --name ${{inputs.name}} \
                                            --input_dir ${{inputs.input_dir}} --model_dir ${{inputs.model_dir}}",
            # add the folder that needs to be mounted as input argument
            # add this as extra argument to val.py (make change accordingly in train.py)
            environment=pipeline_job_env,
            compute="Image-Classifier",
        )

        @dsl.pipeline(
            description="Yolov5 Training Pipeline for AI City challenge",
        )
        def object_detection_pipeline():
            # using data_prep_function like a python call with its own inputs
            data_process = data_process_job()

            # using train_func like a python call with its own inputs
            train_process = training_job(input_dir=data_process.outputs.output_dir)

            # Test performance extraction
            test_process = test_job(input_dir=data_process.outputs.output_dir,
                                    model_dir=train_process.outputs.output_dir)

            # a pipeline returns a dictionary of outputs
            # keys will code for the pipeline output identifier
            return None

        # Instantiate
        pipeline = object_detection_pipeline()

        # submit the pipeline job
        pipeline_job = ml_client.jobs.create_or_update(
            pipeline,
            experiment_name="aicity_detection" # Project's name
        )

        # open the pipeline in web browser
        webbrowser.open(pipeline_job.services["Studio"].endpoint)

    else:
        print("Please pass the argument cloud to trigger a remote ml workflow job")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cloud',
                        action='store_true',
                        help='Flag to indicate whether to run pipeline locally or on cloud')

    # Reading the arguments
    args = parser.parse_args()
    cloud = args.cloud

    run_train_pipeline(cloud)


