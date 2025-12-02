import torch as t
from torch import nn

def l1_loss(y_true, y_pred):
    loss = nn.L1Loss()
    return loss(y_true, y_pred)


def l2_loss(y_true, y_pred):
    loss = nn.MSELoss()
    return loss(y_true, y_pred)


# Todo SSIM Loss

# Todo Sobel Loss

# # Todo is it actually discriminator loss?
# def gan_loss(disc_generated_output):
#     loss = nn.BCELoss  # bce loss (todo why)
#     return loss(t.ones_like(disc_generated_output), disc_generated_output)
#
#
# def discriminator_loss(disc_real_output, disc_generated_output):
#     loss = nn.BCELoss  # bce loss (todo why)
#     real_loss = loss(t.ones_like(disc_real_output), disc_real_output)
#
#     generated_loss = loss(t.zeros_like(
#         disc_generated_output), disc_generated_output)
#
#     total_disc_loss = real_loss + generated_loss
#
#     return total_disc_loss
#
#
# # Using Lambda (todo wtf are they doing here)
# def generator_loss(disc_generated_output, gen_output, target, loss_funcs, l):
#     _gan_loss = gan_loss(disc_generated_output)
#
#     _loss = 0
#     for loss_func in loss_funcs:
#         _loss += loss_func(target, gen_output)
#
#     total_gen_loss = _gan_loss + (l * _loss)
#
#     return total_gen_loss, _gan_loss, _loss
