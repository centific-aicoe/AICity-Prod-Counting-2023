# For more details on this example look at link below
# https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/mnist-hello-world.html#A-more-complete-MNIST-Lightning-Module-Example
import sys
import os
import random

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torch.nn.functional as F


class LitImageSeg(LightningModule):
    def __init__(self, data_dir=None, n_channels=3, learning_rate=2e-4):
        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.n_channels = n_channels
        self.learning_rate = learning_rate

        # Model Architecture needs to be defined
        """ Convolutional block:
            It follows a two 3x3 convolutional layer, each followed by a batch normalization and a relu activation.
        """
        class conv_block(nn.Module):
            def __init__(self, in_c, out_c):
                super().__init__()

                self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm2d(out_c)

                self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm2d(out_c)

                self.relu = nn.ReLU()

            def forward(self, inputs):
                x = self.conv1(inputs)
                x = self.bn1(x)
                x = self.relu(x)

                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu(x)

                return x

        """ Encoder block:
            It consists of an conv_block followed by a max pooling.
            Here the number of filters doubles and the height and width half after every block.
        """

        class encoder_block(nn.Module):
            def __init__(self, in_c, out_c):
                super().__init__()

                self.conv = conv_block(in_c, out_c)
                self.pool = nn.MaxPool2d((2, 2))

            def forward(self, inputs):
                x = self.conv(inputs)
                p = self.pool(x)

                return x, p

        """ Decoder block:
            The decoder block begins with a transpose convolution, followed by a concatenation with the skip
            connection from the encoder block. Next comes the conv_block.
            Here the number filters decreases by half and the height and width doubles.
        """

        class decoder_block(nn.Module):
            def __init__(self, in_c, out_c):
                super().__init__()

                self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
                self.conv = conv_block(out_c + out_c, out_c)

            def forward(self, inputs, skip):
                x = self.up(inputs)
                x = torch.cat([x, skip], axis=1)
                x = self.conv(x)

                return x

        """ Encoder """
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, x):

        """ Encoder """
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ Classifier """
        outputs = self.outputs(d4)

        return outputs

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        # Have a way to compute accuracy

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

