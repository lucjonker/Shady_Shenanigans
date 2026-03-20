import argparse
import os
import time

import numpy as np
import torch

torch.set_float32_matmul_precision('medium')

from pathlib import Path as P, Path
import lightning as L
from torch import optim
from torch.utils.data import DataLoader

import loss_functions
from dataset import DSMShadeDataset
from model import ShadyModel
from utils import Sobel
from lightning.fabric.utilities import AttributeDict

TEST_DATA_PATH = "../../training_data/"
CSV_PATH = "../resources/dataset.csv"
CHECKPOINT_PATH = "../results/checkpoint.ckpt"


# https://reit.pages.ewi.tudelft.nl/course-scalable-ai-101-on-daic/075-handson-pytorch-to-fabric.html

def train(args):
    # Launch fabric
    fabric = L.Fabric(accelerator="cuda",
                      devices=args.devices,
                      strategy="ddp")
    fabric.launch()

    fabric.seed_everything(42 + fabric.global_rank)
    np.random.seed(42)

    # Instantiate the custom dataset
    data_path = P(os.getenv('DATASETS_ROOT', default=TEST_DATA_PATH))
    dataset = DSMShadeDataset(CSV_PATH, data_path, max_cache=100)

    # Creating train and validation datasets:
    dataset_size = len(dataset)
    val_length = np.floor(args.validation_split * dataset_size)
    train_data, validation_data = torch.utils.data.random_split(
        dataset,
        (len(dataset) - val_length, val_length)
    )

    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)

    model = ShadyModel()
    model.setup_models()
    disc_optimizer = optim.Adam(params=model.discriminator.parameters(), lr=args.learning_rate,
                                betas=(args.momentum, 0.999))
    gen_optimizer = optim.Adam(params=model.generator.parameters(), lr=args.learning_rate,
                               betas=(args.momentum, 0.999))

    # Fabric setup
    generator, gen_optimizer = fabric.setup(model.generator, gen_optimizer)
    discriminator, disc_optimizer = fabric.setup(model.discriminator, disc_optimizer)

    train_loader = fabric.setup_dataloaders(train_loader)
    validation_loader = fabric.setup_dataloaders(validation_loader)

    device = fabric.device
    sobel = Sobel(device)
    losses = {'train_g_losses': [], 'train_d_losses': [], 'val_g_losses': [], 'val_d_losses': [], 'time': []}
    state = AttributeDict(generator=generator, discriminator=discriminator, gen_optimizer=gen_optimizer,
                          disc_optimizer=disc_optimizer, losses=losses)

    # If we have a checkpoint
    my_file = Path(CHECKPOINT_PATH)
    if my_file.is_file():
        fabric.load(CHECKPOINT_PATH, state)

    start_time = time.time()
    fabric.print("Beginning Training...")
    for epoch in range(args.epochs):
        model.train()
        train_g_loss, train_d_loss = 0.0, 0.0
        fabric.print(f"Train epoch {epoch}")
        for i, data in enumerate(train_loader, 0):
            source, target = data["source"], data["target"]
            ### DISCRIMINATOR TRAINING LOOP ###
            disc_optimizer.zero_grad()
            y_hat = generator(source)
            discriminator_real = discriminator(source, target)
            discriminator_generated = discriminator(source, y_hat)
            disc_loss = loss_functions.discriminator_loss(discriminator_real, discriminator_generated)
            fabric.backward(disc_loss, retain_graph=True)
            disc_optimizer.step()

            ### GENERATOR TRAINING LOOP ###
            gen_optimizer.zero_grad()
            discriminator_output = discriminator(source, y_hat)
            gen_loss = loss_functions.generator_loss(discriminator_output, y_hat, target, args.LAMBDA, sobel)
            fabric.backward(gen_loss)
            gen_optimizer.step()

            train_d_loss += disc_loss.item()
            train_g_loss += gen_loss.item()

        # VALIDATION
        fabric.print(f"Val epoch {epoch}")
        model.eval()
        val_g_loss, val_d_loss = 0.0, 0.0

        with torch.no_grad():
            for i, data in enumerate(validation_loader, 0):
                source, target = data["source"], data["target"]

                y_hat = model.generator(source)
                real_output = model.discriminator(source, target)
                discriminator_generated = model.discriminator(source, y_hat)

                disc_loss = loss_functions.discriminator_loss(real_output, discriminator_generated)
                gen_loss = loss_functions.generator_loss(discriminator_generated, y_hat, target, args.LAMBDA, sobel)

                val_d_loss += disc_loss.item()
                val_g_loss += gen_loss.item()

        # Only record stats on one process
        if fabric.is_global_zero:
            # average train loss per epoch
            losses['train_d_losses'].append(train_d_loss / len(train_loader))
            losses['train_g_losses'].append(train_g_loss / len(train_loader))

            # average val loss per epoch
            losses['val_d_losses'].append(val_d_loss / len(validation_loader))
            losses['val_g_losses'].append(val_g_loss / len(validation_loader))
            losses['time'].append(time.time() - start_time)

            if epoch % 5 == 0:
                fabric.print(f"Checkpoint epoch {epoch}")
                fabric.save(CHECKPOINT_PATH, state)


def run():
    parser = argparse.ArgumentParser(description="Train a Shade raster prediction model")
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size for training and validation')
    parser.add_argument('--workers', type=int, default=2, help='Number of workers for DataLoader')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Initial learning rate')
    parser.add_argument('--validation_split', type=float, default=.2, help='Ratio of training data for validation')
    parser.add_argument('--momentum', type=float, default=0.5, help='Optimizer momentum')
    parser.add_argument('--LAMBDA', type=float, default=100,
                        help='Determines ratio between adversarial and composite loss')

    parser.add_argument('--devices', type=int, default=3, help='Number of GPUs to use')
    args = parser.parse_args()

    # Run training
    train(args)


# Inspiration: https://reit.pages.ewi.tudelft.nl/course-scalable-ai-101-on-daic/075-handson-pytorch-to-fabric.html
if __name__ == '__main__':
    run()
