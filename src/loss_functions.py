import torch
from torch import nn
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics import MeanSquaredError


# Loss definitions inspired by https://github.com/uic-evl/deep-umbra/blob/main/deep_shadow.py
def l1_loss(y_true, y_pred):
    loss = nn.L1Loss()
    return loss(y_pred, y_true)


def bce_loss(y_pred, y_true):
    loss = nn.BCELoss()
    return loss(y_pred, y_true)


def ssim(y_true, y_pred):
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(y_pred.device)
    return ssim(y_pred, y_true)

def rmse(y_true, y_pred):
    mse = MeanSquaredError().to(y_pred.device)
    mse = mse(y_pred, y_true)
    return torch.sqrt(mse)


def ssim_loss(y_true, y_pred):
    val = ssim(y_true, y_pred)
    return 1 - val


def sobel_loss(sobel, y_true, y_pred):
    square = torch.square(sobel(y_true) - sobel(y_pred))
    # 2D to 1D
    m1 = torch.mean(square)
    # 1D to 1 Value
    return torch.mean(m1)


def composite_loss(sobel, y_true, y_pred):
    l1loss = l1_loss(y_true, y_pred)
    ssimloss = ssim_loss(y_true, y_pred)
    sobelloss = sobel_loss(sobel, y_true, y_pred)
    return l1loss + ssimloss + sobelloss


def generator_loss(discriminator_output, y_hat, target, loss_lambda, sobel):
    real_class = torch.ones_like(discriminator_output, device=discriminator_output.device)
    adversarial_loss = bce_loss(discriminator_output, real_class)
    composite = composite_loss(sobel, y_hat, target)

    total_loss = adversarial_loss + (loss_lambda * composite)
    return total_loss


def discriminator_loss(discriminator_real, discriminator_generated):
    real_class = torch.ones_like(discriminator_real, device=discriminator_real.device)
    real_loss = bce_loss(discriminator_real, real_class)

    fake_class = torch.zeros_like(discriminator_generated, device=discriminator_generated.device)
    fake_loss = bce_loss(discriminator_generated, fake_class)

    total_loss = (real_loss + fake_loss)
    return total_loss
