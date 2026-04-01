import argparse
import os
import time

import numpy as np
import torch
from lightning.pytorch.plugins import TorchSyncBatchNorm
from lightning_fabric.loggers import CSVLogger

torch.set_float32_matmul_precision('medium')
torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

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
SAVE_FREQUENCY = 5


# https://reit.pages.ewi.tudelft.nl/course-scalable-ai-101-on-daic/075-handson-pytorch-to-fabric.html

def train(args):
    logger = CSVLogger("../results/", "loss_logs", flush_logs_every_n_steps=SAVE_FREQUENCY)
    # Launch fabric
    fabric = L.Fabric(accelerator="cuda",
                      devices=args.devices,
                      strategy="ddp",
                      loggers=logger)
    fabric.launch()
    fabric.seed_everything(42 + fabric.global_rank)

    # Instantiate the custom dataset
    data_path = P(os.getenv('DATASETS_ROOT', default=TEST_DATA_PATH))
    dataset = DSMShadeDataset(CSV_PATH, data_path, max_cache=100)

    # Creating train and validation datasets:
    dataset_size = len(dataset)
    val_length = np.floor(args.validation_split * dataset_size)
    train_data, validation_data = torch.utils.data.random_split(
        dataset,
        (int(len(dataset) - val_length), int(val_length)),
    )

    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, num_workers=args.workers)

    print(f"[Rank {fabric.global_rank}] train len={len(train_loader)}, val len={len(validation_loader)}")

    with fabric.init_module():
        model = ShadyModel()

    model.setup_initial_weights()

    disc_optimizer = optim.Adam(params=model.discriminator.parameters(), lr=args.learning_rate,
                                betas=(args.momentum, 0.999))
    gen_optimizer = optim.Adam(params=model.generator.parameters(), lr=args.learning_rate,
                               betas=(args.momentum, 0.999))

    # Fabric setup
    batch_sync = TorchSyncBatchNorm()
    synced_generator = batch_sync.apply(model.generator)
    synced_discriminator = batch_sync.apply(model.discriminator)
    generator, gen_optimizer = fabric.setup(synced_generator, gen_optimizer)
    discriminator, disc_optimizer = fabric.setup(synced_discriminator, disc_optimizer)
    train_loader, validation_loader = fabric.setup_dataloaders(train_loader, validation_loader)
    is_distributed = torch.distributed.is_initialized()

    device = fabric.device
    sobel = Sobel(device)

    # Prepare logging variables
    epoch = 0
    state = AttributeDict(generator=generator, discriminator=discriminator, gen_optimizer=gen_optimizer,
                          disc_optimizer=disc_optimizer, epoch=epoch, train_height_max=dataset.training_max)
    # If we have a checkpoint
    my_file = Path(CHECKPOINT_PATH)
    if my_file.is_file():
        fabric.print("Loading checkpoint")
        fabric.load(CHECKPOINT_PATH, state)

    start_time = time.time()
    fabric.print("Beginning Training...")
    for e in range(epoch, args.epochs):
        epoch = e
        model.train()
        train_g_loss, train_d_loss = 0.0, 0.0
        print(f"[Rank {fabric.global_rank}] starting training {epoch}")
        for i, data in enumerate(train_loader, 0):
            source, target = data["source"], data["target"]
            ### DISCRIMINATOR TRAINING LOOP ###
            disc_optimizer.zero_grad()
            y_hat = generator(source).detach()
            discriminator_real = discriminator(source, target)
            discriminator_generated = discriminator(source, y_hat)
            disc_loss = loss_functions.discriminator_loss(discriminator_real, discriminator_generated)
            fabric.backward(disc_loss)
            disc_optimizer.step()

            ### GENERATOR TRAINING LOOP ###
            gen_optimizer.zero_grad()
            y_hat = generator(source)
            discriminator_output = discriminator(source, y_hat)
            gen_loss = loss_functions.generator_loss(discriminator_output, y_hat, target, args.LAMBDA, sobel)
            fabric.backward(gen_loss)
            gen_optimizer.step()

            train_d_loss = train_d_loss + disc_loss.item()
            train_g_loss = train_g_loss + gen_loss.item()

        # Aggregate training losses
        d_tensor = torch.tensor(train_d_loss, device=fabric.device, dtype=torch.float32)
        g_tensor = torch.tensor(train_g_loss, device=fabric.device, dtype=torch.float32)
        train_d_loss, train_g_loss = aggregate_loss(fabric, d_tensor, g_tensor, is_distributed,
                                                    len(train_loader))

        # VALIDATION
        print(f"[Rank {fabric.global_rank}] starting validation {epoch}")
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

                val_d_loss = val_d_loss + disc_loss.item()
                val_g_loss = val_g_loss + gen_loss.item()

        # Aggregate validation losses
        d_tensor = torch.tensor(val_d_loss, device=fabric.device, dtype=torch.float32)
        g_tensor = torch.tensor(val_g_loss, device=fabric.device, dtype=torch.float32)
        val_d_loss, val_g_loss = aggregate_loss(fabric, d_tensor, g_tensor, is_distributed,
                                                len(validation_loader))

        # Only record stats on one process
        if fabric.global_rank == 0:
            losses = {'train_g_losses': train_g_loss,
                      'train_d_losses': train_d_loss,
                      'val_g_losses': val_g_loss,
                      'val_d_losses': val_d_loss,
                      'time': time.time() - start_time}
            fabric.log_dict(losses, epoch)

        if (epoch + 1) % SAVE_FREQUENCY == 0:
            fabric.print(f"Checkpoint epoch {epoch}")
            # Automatically runs on rank 0
            fabric.save(CHECKPOINT_PATH, state)

        print(f"[Rank {fabric.global_rank}] finished epoch {epoch}")


def aggregate_loss(fabric, d_loss, g_loss, is_distributed, num_samples):
    if is_distributed:
        # Take the mean over the different GPUS
        d_loss = fabric.all_reduce(d_loss, reduce_op="mean").item()
        g_loss = fabric.all_reduce(g_loss, reduce_op="mean").item()

    # Get approximate per-item loss
    d_loss /= num_samples
    g_loss /= num_samples

    return d_loss, g_loss


def run():
    parser = argparse.ArgumentParser(description="Train a Shade raster prediction model")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
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
