
import io
from datetime import datetime
from typing import List, Tuple

import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch import nn
from torch.nn import functional as F


def load_images_from_indices(testset, indices, device="mps"):
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # Pick 30 samples randomly
    raw_images = [np.asarray(testset[i][0]) for i in indices]
    # Convert the images to tensors
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    images = torch.stack([test_transform(image) for image in raw_images])
    # Move the images to the device
    images = images.to(device)
    return images


def visualize_attention(attention_map: torch.Tensor) -> Image.Image:
    """
    Plots the attention maps as a grid of heatmaps. Each column shows a head.

    Args:
        attention_map: Tensor of shape (batch, num_heads, seq_len, seq_len)
    Returns:
        PIL Image with the full attention grid.
    """
    attn = attention_map.detach().cpu().float().numpy()
    batch_size, num_heads, seq_len, _ = attn.shape

    fig = plt.figure(figsize=(2.5 * num_heads, 2.5 * batch_size))
    gs = gridspec.GridSpec(
        batch_size, num_heads,
        figure=fig,
        hspace=0.4,
        wspace=0.3
    )

    vmin = attn.min()
    vmax = attn.max()

    for b in range(batch_size):
        for h in range(num_heads):
            ax = fig.add_subplot(gs[b, h])
            im = ax.imshow(
                attn[b, h],
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
                interpolation="nearest"
            )
            ax.set_title(f"head {h}", fontsize=9, pad=4)
            ax.set_ylabel(f"batch {b}" if h == 0 else "", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    # Colorbar compartida
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    image = Image.open(buf).copy()  # .copy() para cerrar el buffer sin perder la imagen
    plt.close(fig)

    return image


def visualize_attention_on_images(
    attention_maps,
    images,
    alpha: float = 0.6,
    colormap=cm.viridis,
    token_idx: int = 0,
    figsize_per_cell: tuple = (2, 2),
    dpi: int = 150,
):
    if isinstance(attention_maps, torch.Tensor):
        attention_maps = [attention_maps]

    result_images = []

    imgs_np = (images.detach().cpu().float() * 0.5 + 0.5).clamp(0, 1)
    imgs_np = imgs_np.permute(0, 2, 3, 1).numpy()
    img_h, img_w = imgs_np.shape[1], imgs_np.shape[2]

    for block_idx, amap in enumerate(attention_maps):
        amap = amap.detach().cpu().float()
        assert amap.ndim == 4, f"Esperado [B, H, N, N], got {amap.shape}"
        B, H, N_q, N_k = amap.shape

        has_cls = N_k > 1
        if has_cls:
            attn = amap[:, :, token_idx, 1:]   # [B, H, n_img_tokens]
        else:
            attn = amap[:, :, token_idx, :]

        patch_count = attn.shape[-1]
        patch_grid_side = int(patch_count ** 0.5)

        if patch_grid_side ** 2 != patch_count:
            raise ValueError(
                f"Bloque {block_idx}: {patch_count} tokens no forman grid cuadrado."
            )

        n_rows = B
        n_cols = 1 + H
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
            squeeze=False
        )
        fig.suptitle(
            f"Block {block_idx}  |  {H} heads  |  {patch_grid_side}×{patch_grid_side} patches",
            fontsize=11
        )

        axes[0, 0].set_title("original", fontsize=8)
        for h in range(H):
            axes[0, h + 1].set_title(f"head {h}", fontsize=8)

        for b in range(B):
            img = imgs_np[b]

            axes[b, 0].imshow(img.clip(0, 1))
            axes[b, 0].axis("off")
            axes[b, 0].set_ylabel(f"img {b}", fontsize=7)

            for h in range(H):
                a = attn[b, h].numpy()   # [patch_count]

                # Normalización robusta: percentil en lugar de min/max
                # evita que un outlier aplaste toda la escala
                lo, hi = np.percentile(a, 5), np.percentile(a, 95)
                a_norm = np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)
                a_norm = a_norm.reshape(patch_grid_side, patch_grid_side)

                # Resize con BICUBIC para bordes más suaves
                a_resized = np.array(
                    Image.fromarray((a_norm * 255).astype(np.uint8)).resize(
                        (img_w, img_h), resample=Image.BICUBIC
                    )
                ) / 255.0

                # Overlay: imagen base + heatmap con alpha solo en el canal de color
                heatmap_rgba = colormap(a_resized)          # [H, W, 4]
                heatmap_rgb = heatmap_rgba[..., :3]         # [H, W, 3]

                # Blend aditivo ponderado por la intensidad de atención
                # — donde hay poca atención, se ve más la imagen original
                weight = a_resized[..., np.newaxis]         # [H, W, 1], en [0,1]
                overlay = (img * (1 - alpha * weight) + heatmap_rgb * alpha * weight).clip(0, 1)

                axes[b, h + 1].imshow(overlay)
                axes[b, h + 1].axis("off")

        plt.tight_layout()

        sm = cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        result_images.append(Image.open(buf).copy())
        plt.close(fig)

    return result_images