import torch.nn as nn

from src import loss_functions
from src.dataset import TARGET_W, TARGET_H


# Todo Extend
class ShadyModel:
    def __init__(self, ngpu, device, tile_size=512):
        self.ngpu = ngpu
        self.device = device
        self.generator = Generator(ngpu, tile_size)
        # Todo replace static loss with implemented discriminator
        self.discriminator = loss_functions.l1_loss

    def setup_models(self):
        # # Handle multi-GPU if desired (todo add mac version?)
        # if (self.device.type == 'cuda') and (self.ngpu > 1):
        #     self.generator = nn.DataParallel(self.generator, list(range(self.ngpu)))
        #     # self.discriminator = nn.DataParallel(self.discriminator, list(range(self.ngpu)))
        self.generator.to(self.device)
        # Apply the weights_init function to randomly initialize
        # all weights to mean=0, stdev=0.02
        self.generator.apply(weights_init)
        # self.discriminator.apply(weights_init)

class Generator(nn.Module):
    def __init__(self, ngpu, tile_size: int):
        super(Generator, self).__init__()
        self.tile_size_w = tile_size if tile_size is not None else TARGET_W
        self.tile_size_h = tile_size if tile_size is not None else TARGET_H
        self.dim_in = 5 * self.tile_size_w * self.tile_size_h
        self.dim_out = self.tile_size_w * self.tile_size_h
        # Todo: Incredibly basic network for testing
        self.network = nn.Sequential(
            nn.Linear(self.dim_in, 10),
            nn.ReLU(),
            nn.Linear(10, self.dim_out)
        )
        self.ngpu = ngpu

    def forward(self, x):
        # super basic linear pass for testing
        x = x.view(-1, self.dim_in)
        y = self.network(x)
        # Todo: Incredibly basic network for testing
        return y.view(1, 1, self.tile_size_h, self.tile_size_w)


# # Todo Define Discriminator
# class Discriminator(nn.Module):
#     def __init__(self, ngpu):
#         super(Discriminator, self).__init__()
#         self.ngpu = ngpu
#
#     def forward(self, x):
#         return self.main(x)


# Todo: do I want this at all? https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
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
