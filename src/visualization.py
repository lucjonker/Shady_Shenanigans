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


# Plots result and target images assuming they come from the gpu
def display_res(source, result, target):
    source_cpu = source.cpu()
    result_cpu = result.cpu().detach()
    target_cpu = target.cpu().detach()

    plt.figure(figsize=(16, 6))
    ax1 = plt.subplot(1, 3, 1)
    result = source_cpu[0][0]
    plt.imshow(result.numpy())
    ax1.title.set_text("Source")
    plt.axis('off')

    ax2 = plt.subplot(1, 3, 2)
    result = result_cpu[0]
    plt.imshow(result.squeeze().numpy())
    ax2.title.set_text("Model Result")
    plt.axis('off')

    ax3 = plt.subplot(1, 3, 3)
    result = target_cpu[0]
    plt.imshow(result.squeeze().numpy())
    ax3.title.set_text("Target Result")
    plt.axis('off')

    plt.set_cmap("viridis")
    plt.show()