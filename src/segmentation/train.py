import os
import sys
import torch
import mlflow
import argparse
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

# Adding working directory to path to run the files as standalone
curr_wd = os.getcwd()
if curr_wd not in sys.path:
    sys.path.append(curr_wd)  # add working directory to PATH

# Importing Data Module and Model
from src.segmentation.datamodule import SegDataModule
from src.segmentation.model import LitImageSeg

# Load environment variables
# These lines are to load tracking server url and connection strings to log metrics and artifacts
from dotenv import load_dotenv
load_dotenv()


def train_model(data_dir, learning_rate, epochs, output_folder):

    # The data_dir folder will have two subfolders
    # 1. train - images
    # 2. segmentation_labels - masks

    # Data Module Loading
    BATCH_SIZE = 32 if torch.cuda.is_available() else 8
    print("batch size", BATCH_SIZE)
    print("gpu thing", torch.cuda.is_available())

    torch_version = torch.__version__
    print("torch version is ", torch_version)

    seg_dm = SegDataModule(data_dir=data_dir, batch_size=BATCH_SIZE)

    # Define call backs
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_folder, monitor="val_loss", mode="min", filename='image_segmentation_model'
    )

    # Pytorch lightning module
    model = LitImageSeg(data_dir=data_dir, learning_rate=learning_rate)

    # Create trainer
    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=epochs,
        callbacks=[TQDMProgressBar(refresh_rate=20), checkpoint_callback],
        logger=CSVLogger(save_dir="./logs"),
    )

    print("Environment MLFLOW TRACKING VARIABLE", os.environ['MLFLOW_TRACKING_URI'])
    print('MLFLOW TRACKING URI ------------->', mlflow.get_tracking_uri())

    with mlflow.start_run() as run:

        # Train Model
        trainer.fit(model, datamodule=seg_dm)

        # Extract performance on test set
        trainer.test(datamodule=seg_dm)

        # Get all the info of train, val and test images for future re-use
        lst_train_images = [{"image_name": os.path.basename(item), "tag": "train"}
                            for item in seg_dm.train_dataset.imagePaths]
        lst_val_images = [{"image_name": os.path.basename(item), "tag": "val"}
                          for item in seg_dm.val_dataset.imagePaths]
        lst_test_images = [{"image_name": os.path.basename(item), "tag": "test"}
                           for item in seg_dm.test_dataset.imagePaths]

        # get a dataframe
        dataset_info = pd.concat([pd.DataFrame(lst_train_images), pd.DataFrame(lst_val_images),
                                 pd.DataFrame(lst_test_images)], axis=0)
        dataset_info_path = os.path.join(output_folder, "dataset_info.csv")
        dataset_info.to_csv(dataset_info_path, index=False)

        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch size", BATCH_SIZE)
        mlflow.log_param("learning rate", learning_rate)

        metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
        del metrics["step"]
        metrics.set_index("epoch", inplace=True)
        sn.relplot(data=metrics, kind="line")

        plot_filename = os.path.join(output_folder, "loss_plot.png")
        plt.savefig(plot_filename)

        # Saving model
        model_out_path = os.path.join("outputs", "image_segmentation_model.pth")
        torch.save(model.state_dict(), model_out_path)

        # Logging artifacts
        mlflow.log_artifact(f"{trainer.logger.log_dir}/metrics.csv")
        mlflow.log_artifact(plot_filename)
        mlflow.log_artifact(dataset_info_path)
        mlflow.log_artifact(trainer.ckpt_path)
        mlflow.log_artifact(model_out_path)

        test_metrics = trainer.callback_metrics
        print(test_metrics)

        for metric_name in test_metrics:
            mlflow.log_metric(metric_name, test_metrics[metric_name])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',
                        help='Input Directory for raw data',
                        required=True,
                        type=str)
    parser.add_argument('--learning_rate',
                        help='Learning rate for the model',
                        required=True,
                        type=float)
    parser.add_argument('--epochs',
                        help='Number of epochs to be trained',
                        required=True,
                        type=int)
    parser.add_argument('--output_folder',
                        help='Directory to save model outputs data',
                        required=True,
                        type=str)

    # Reading the arguments
    args = parser.parse_args()

    # Reading the arguments
    data_dir = args.data_dir
    output_folder = args.output_folder
    learning_rate = args.learning_rate
    epochs = args.epochs

    train_model(data_dir, learning_rate, epochs, output_folder)









