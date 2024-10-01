import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import wandb
from typing import Tuple, Iterable


def create_rgb_with_masks(
    rgbs: Tuple[Iterable[np.ndarray], np.ndarray],
    segs: Tuple[Iterable[np.ndarray], np.ndarray],
) -> wandb.Image:
    current_backend = matplotlib.get_backend()
    matplotlib.use("agg")

    if isinstance(rgbs, Iterable):
        rgbs = np.array(rgbs)
    if isinstance(segs, Iterable):
        segs = np.array(segs)

    if rgbs.ndim == 3:
        rgbs = rgbs[None, ...]

    if segs.ndim == 3 and segs.shape[-1] == 1:
        segs = segs[None, ..., 0]
    elif segs.ndim == 2:
        segs = segs[None, ...]

    assert rgbs.ndim == 4 and rgbs.shape[-1] == 3 and segs.ndim == 3

    N_images = rgbs.shape[0]

    fig, axes = plt.subplots(nrows=2, ncols=N_images, squeeze=False)
    for ax in axes.flatten():
        ax.axis("off")

    for n in range(N_images):
        axes[0][n].imshow(rgbs[n])
        axes[1][n].imshow(segs[n], vmin=0, vmax=1)
    fig.tight_layout()

    wandb_img = wandb.Image(fig)

    plt.close(fig)

    matplotlib.use(current_backend)

    return wandb_img
