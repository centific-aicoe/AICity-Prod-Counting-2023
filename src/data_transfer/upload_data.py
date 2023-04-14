import os
import sys
import time
import argparse
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
load_dotenv()

# Adding working directory to path to run the files as standalone
curr_wd = os.getcwd()
if curr_wd not in sys.path:
    sys.path.append(curr_wd)  # add working directory to PATH

from src.utils.helpers import fancy_print


def upload_folder_to_azure_container(connection_string, container_name, local_fold_path):
    """
    Upload contents of a local folder to a blob container
    :param connection_string: connection string to access the azure blob container
    :param container_name: Name of the container
    :param local_fold_path: Path of the local directory which will be uploaded to blob
    """

    # Initialize the connection to Azure storage account
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Base folder name
    base_folder_name = os.path.basename(local_fold_path)

    fancy_print(f"Uploading folder {base_folder_name} from {local_fold_path}")
    print(f"Azure Container: {container_name}")

    # Initializing a file counter
    uploaded_file_count = 0

    # Iterate over the folder and upload folders and files recursively
    for root, dirs, files in os.walk(local_fold_path):
        for filename in files:
            file_path_on_azure = os.path.join(root, filename).replace(local_fold_path, base_folder_name)
            file_path_on_local = os.path.join(root, filename)

            if filename == ".DS_Store":
                continue

            try:
                blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_path_on_azure)

                # If you want files to be overwritten, add argument overwrite=True to the upload_blob method
                with open(file_path_on_local, "rb") as data:
                    blob_client.upload_blob(data)

                # Updating file count
                uploaded_file_count += 1

            except Exception as e:
                print(str(e))

    print(f"Total number of files uploaded: {uploaded_file_count}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_fold_path',
                        help='Path of the folder which needs to be uploaded to blob container',
                        required=True,
                        type=str)

    parser.add_argument('--container_name',
                        help='the container under which data is stored in azure',
                        required=True,
                        type=str)

    args = parser.parse_args()

    # Reading the arguments
    local_fold_path = args.local_fold_path
    container_name = args.container_name

    # Getting details for the server from environment variables
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

    if connection_string is None:
        print("Required Environment variable AZURE_STORAGE_CONNECTION_STRING not found")

    else:
        # Timer Begin
        start = time.time()

        upload_folder_to_azure_container(connection_string, container_name, local_fold_path)

        # Timer End
        end = time.time()

        # Computing time taken
        time_taken_mins = round((end - start) / 60, 1)

        print("Time taken for uploading data is {} minutes".format(time_taken_mins))

