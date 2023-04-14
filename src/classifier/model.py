
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torch.nn.functional as F
import torchvision.models as models
from torchmetrics import Accuracy


class ResNetProdClassifier(LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3, transfer=True, tune_fc_only=True):
        super().__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.transfer = transfer
        self.tune_fc_only = tune_fc_only

        # Model architecture
        self.resnet_model = models.resnet18(pretrained=True)

        # Get last layer of resnet
        last_layer = list(self.resnet_model.children())[-1]
        last_layer_in_features = last_layer.in_features

        # replace final layer for fine tuning
        self.resnet_model.fc = nn.Linear(in_features=last_layer_in_features,
                                         out_features=self.num_classes)

        if tune_fc_only:  # option to only tune the fully-connected layers
            for child in list(self.resnet_model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # metrics
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.resnet_model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)

        # Computing accuracy
        self.val_accuracy.update(preds, y)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)

        # Computing accuracy
        self.test_accuracy.update(preds, y)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == "__main__":
    model = ResNetProdClassifier(num_classes=116)
    test_input = torch.randn((1, 3, 224, 224))
    test_output = model(test_input)
    print(test_output.shape)