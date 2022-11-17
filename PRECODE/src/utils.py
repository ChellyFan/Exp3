import numpy as np
import matplotlib.pyplot as plt
import torch

def get_mean_std(dataset, ratio=0.01):
    """Get mean and std by sample ratio
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(len(dataset) * ratio),
                                             shuffle=True, num_workers=0)
    train = iter(dataloader).next()[0]
    mean = np.mean(train.numpy(), axis=(0, 2, 3))
    std = np.std(train.numpy(), axis=(0, 2, 3))
    return mean, std


class DeNormalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor.mul(self.std).add(self.mean)


def show_reconstructions(images, dm, ds):
    fig=plt.figure(figsize=(10,5), dpi=150)
    columns = len(images)
    rows = 1

    denorm = DeNormalizer(dm, ds)
    for i, image in enumerate(images):
        ax = fig.add_subplot(rows, columns, i+1)
        plt.axis('off')
        image = denorm(image).permute((1,2,0)).cpu().numpy().squeeze()
        if len(image.shape) < 3:
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image)

    axes = fig.axes
    for ax, title in zip(axes, ['Original', 'IGA&Base', 'IGA&PRECODE', 'NRA&PRECODE']):
        ax.set_title(title)

    fig.tight_layout()
    plt.show()
    plt.close()