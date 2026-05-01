import numpy as np
import rasterio
from matplotlib import pyplot as plt


def plot_losses(g_losses, d_losses, title="Loss Analysis:"):
    plt.figure(figsize=(10, 5))
    plt.suptitle(title, fontsize=16)

    plt.plot(range(1, len(g_losses) + 1), g_losses, label="Generator Loss")
    plt.plot(range(1, len(d_losses) + 1), d_losses, label="Discriminator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Generator and Discriminator Loss Over Epochs")
    plt.legend()
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