import numpy as np
import rasterio
import pandas as pd
from matplotlib import pyplot as plt

import seaborn as sns
sns.set_theme()
sns.set_context("paper")

def plot_losses(g_train, d_train, g_val, d_val):
    sns.relplot(
        data=(g_train, d_train, g_val, d_val), kind="line", palette="colorblind", aspect=1.5
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()


# Plots result and target images
def display_res(source, generated, target):
    source = load_raster(source)
    generated = load_raster(generated)
    target = load_raster(target)

    plt.figure(figsize=(16, 6))
    ax1 = plt.subplot(1, 3, 1)
    plt.imshow(source)
    ax1.title.set_text("Source")
    plt.axis('off')

    ax2 = plt.subplot(1, 3, 2)
    plt.imshow(generated)
    ax2.title.set_text("Model Result")
    plt.axis('off')

    ax3 = plt.subplot(1, 3, 3)
    plt.imshow(target)
    ax3.title.set_text("Target Result")
    plt.axis('off')

    plt.set_cmap("viridis")
    plt.show()


def load_raster(source):
    with rasterio.open(source) as src:
        arr = src.read(1)  # (H, W)
    source = arr.astype(np.float32)
    return source