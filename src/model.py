# Inspiration for this model implementation:
# https://medium.com/@ms.maryamrezaee/pix2pix-pytorch-implementation-what-is-it-and-how-to-do-it-f53bce51c84e
# https://neptune.ai/blog/pix2pix-key-model-architecture-decisions
# https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/

import os

import torch
import torch.nn as nn


# Todo Extend
class ShadyModel:
    def __init__(self):
        # Tile size hard coded for 512x512
        self.generator = Generator()
        self.discriminator = Discriminator()

    def setup_initial_weights(self, eval_only: bool = False):
        self.generator.apply(weights_init)
        if not eval_only:
            self.discriminator.apply(weights_init)

    def train(self):
        self.generator.train()
        self.discriminator.train()

    def eval(self):
        self.generator.eval()
        self.discriminator.eval()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Encoder (DownSampling)
        self.down1 = DownSample(6, 64, apply_batchnorm=False)  # C64
        self.down2 = DownSample(64, 128)  # C128
        self.down3 = DownSample(128, 256)  # C256
        self.down4 = DownSample(256, 512)  # C512
        self.down5 = DownSample(512, 512)  # C512
        self.down6 = DownSample(512, 512)  # C512
        self.down7 = DownSample(512, 512)  # C512
        self.down8 = DownSample(512, 512)  # C512
        self.down9 = DownSample(512, 512)  # C512

        # Decoder (Upsampling)
        self.up1 = UpSample(512, 512, apply_dropout=True)  # CD1024
        self.up2 = UpSample(1024, 512, apply_dropout=True)  # CD1024
        self.up3 = UpSample(1024, 512, apply_dropout=True)  # CD1024
        self.up4 = UpSample(1024, 512)  # C1024
        self.up5 = UpSample(1024, 512)  # C1024
        self.up6 = UpSample(1024, 256)  # C1024
        self.up7 = UpSample(512, 128)  # C512
        self.up8 = UpSample(256, 64)  # C256

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Shade output between 0 and 1
        )

    def forward(self, x):
        # Encoder forward      (batch_size, 5, 512, 512)
        d1 = self.down1(x)  # (batch_size, 64, 256, 256)
        d2 = self.down2(d1)  # (batch_size, 128, 128, 128)
        d3 = self.down3(d2)  # (batch_size, 256, 64, 64)
        d4 = self.down4(d3)  # (batch_size, 512, 32, 32)
        d5 = self.down5(d4)  # (batch_size, 512, 16, 16)
        d6 = self.down6(d5)  # (batch_size, 512, 8, 8)
        d7 = self.down7(d6)  # (batch_size, 512, 4, 4)
        d8 = self.down8(d7)  # (batch_size, 512, 2, 2)
        d9 = self.down9(d8)  # (batch_size, 512, 1, 1)

        # Decoder forward + skip connections (U-Net)
        u1 = self.up1(d9, d8)  # (batch_size, 1024, 2, 2)
        u2 = self.up2(u1, d7)  # (batch_size, 1024, 4, 4)
        u3 = self.up3(u2, d6)  # (batch_size, 1024, 8, 8)
        u4 = self.up4(u3, d5)  # (batch_size, 1024, 16, 16)
        u5 = self.up5(u4, d4)  # (batch_size, 1024, 32, 32)
        u6 = self.up6(u5, d3)  # (batch_size, 512, 64, 64)
        u7 = self.up7(u6, d2)  # (batch_size, 256, 128, 128)
        u8 = self.up8(u7, d1)  # (batch_size, 128, 256, 256)
        fin = self.final(u8)  # (batch_size, 1, 512, 512)

        return fin


class DownSample(nn.Module):

    def __init__(self, in_channels, out_channels, apply_batchnorm=True):
        super(DownSample, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not apply_batchnorm)
        ]
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        return self.down(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, apply_dropout=False):
        super(UpSample, self).__init__()

        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        if apply_dropout:
            layers.append(nn.Dropout(0.5))

        self.up = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)  # skip connection concatenating
        return x


class Discriminator(nn.Module):

    # in channels is channels of source + either real or generated image, so 6 for input and 1 for output
    def __init__(self, in_channels=7):
        super(Discriminator, self).__init__()

        def conv_block(in_c, out_c, stride):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2)
            )

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),  # C64, no BatchNorm
            nn.LeakyReLU(0.2),

            conv_block(64, 128, stride=2),  # C128
            conv_block(128, 256, stride=2),  # C256
            conv_block(256, 512, stride=1),  # C512

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),  # Final layer
            nn.Sigmoid()
        )

    def forward(self, x, y):
        concatenated = torch.cat([x, y], dim=1)
        verdict = self.model(concatenated)
        return verdict


# INSPO https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# takes an initialized model as input and reinitializes all convolutional,
# convolutional-transpose, and batch normalization layers to meet the criteria
# that all model weights shall be randomly initialized from a Normal distribution with mean=0, stdev=0.02
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
