import os
import sys
import argparse
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.ai.ml import command
from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes

from azure.identity import DefaultAzureCredential

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
        custom_env_name = "aicity_challenge"
        pipeline_job_env = Environment(
            name=custom_env_name,
            description="Custom environment for AI City challenge",
            conda_file="environment.yml",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.0.3-cudnn8-ubuntu18.04",
        )
        ml_client.environments.create_or_update(pipeline_job_env)

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
        job = command(
            experiment_name="aicity_image_classifier",
            inputs={
                "data_dir": Input(type=AssetTypes.URI_FOLDER,
                                  path="azureml://datastores/aicity_full_data/paths/"),
                # "data_dir": Input(type=AssetTypes.URI_FOLDER,
                #                     path="./datasets"),
                "epochs": 100,
                "learning_rate": 0.002,
                "num_classes": 116
            },
            outputs={
                "output_folder": Output(type=AssetTypes.URI_FOLDER)
            },
            code=".",
            command="python src/classifier/train.py --data_dir ${{inputs.data_dir}} \
                                --epochs ${{inputs.epochs}} --learning_rate ${{inputs.learning_rate}} \
                                --output_folder ${{outputs.output_folder}} --num_classes ${{inputs.num_classes}}",
            environment=pipeline_job_env,
            compute="GPU-machine-T4",
        )

        # submit the command
        returned_job = ml_client.jobs.create_or_update(job)

        # get a URL for the status of the job
        print(returned_job.services["Studio"].endpoint)

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
