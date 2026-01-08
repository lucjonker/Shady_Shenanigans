import torch.nn as nn

import loss_functions


# Todo Define Generator
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

    def forward(self, x):
        return x


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


# Todo Extend
class ShadyModel:
    def __init__(self, ngpu, device):
        self.ngpu = ngpu
        self.device = device
        self.generator = Generator(ngpu).to(self.device)
        # Todo implement discriminator
        self.discriminator = loss_functions.l1_loss

    def setup_models(self):
        # Handle multi-GPU if desired
        if (self.device.type == 'cuda') and (self.ngpu > 1):
            self.generator = nn.DataParallel(self.generator, list(range(self.ngpu)))
            # self.discriminator = nn.DataParallel(self.discriminator, list(range(self.ngpu)))

        # Apply the weights_init function to randomly initialize
        # all weights to mean=0, stdev=0.02
        self.generator.apply(weights_init)
        # self.discriminator.apply(weights_init)
