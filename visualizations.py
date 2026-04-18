
import io
import json
import math
import os
from datetime import datetime

import matplotlib.cm as cm
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


def visualize_attention(attention_maps, device="mps"):
    to_pil = torchvision.transforms.ToPILImage()
    images = []

    for idx, amap in enumerate(attention_maps):
        rows, cols = amap.shape[1], amap.shape[0]
        
        amap_norm = amap - amap.min()
        amap_norm = amap_norm / (amap_norm.max() + 1e-8)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        fig.suptitle(f"Attention Map {idx}", fontsize=14)
        
        for row in range(rows):
            for col in range(cols):
                img_tensor = amap_norm[col, row]
                axes[row, col].imshow(img_tensor.detach().cpu().numpy(), cmap="viridis")
                axes[row, col].axis("off")
                axes[row, col].set_title(f"[{col},{row}]", fontsize=7)
        
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        images.append(Image.open(buf).copy())
        plt.close(fig)

    return images

def visualize_attention_on_images(attention_maps, images, alpha=0.5):
    """
    attention_maps: lista de tensores, cada uno [10, 4, 65, 65]
    images:         Tensor [10, 3, 32, 32]
    """
    patch_grid = int(64 ** 0.5)  # 8
    img_size = images.shape[-1]  # 32
    result_images = []

    imgs_np = (images.cpu().float() * 0.5 + 0.5).clamp(0, 1)
    imgs_np = imgs_np.permute(0, 2, 3, 1).numpy()  # [10, 32, 32, 3]

    for idx, amap in enumerate(attention_maps):
        # Extraer patches: [10, 4, 65, 65] → [:, :, 0, 1:] → [10, 4, 64]
        attn = amap[:, :, 0, 1:]
        n_cols = attn.shape[0]  # 10
        n_rows = attn.shape[1]  # 4 heads

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        fig.suptitle(f"Attention Map {idx}", fontsize=14)

        for row in range(n_rows):
            for col in range(n_cols):
                a = attn[col, row].cpu().float()
                a = a - a.min()
                a = a / (a.max() + 1e-8)
                a = a.reshape(patch_grid, patch_grid).detach().numpy()
                a_resized = np.array(
                    Image.fromarray((a * 255).astype(np.uint8)).resize(
                        (img_size, img_size), resample=Image.BILINEAR
                    )
                ) / 255.0

                heatmap = cm.viridis(a_resized)[..., :3]
                overlay = (1 - alpha) * imgs_np[col] + alpha * heatmap

                axes[row, col].imshow(overlay.clip(0, 1))
                axes[row, col].axis("off")
                if row == 0:
                    axes[row, col].set_title(f"img {col}", fontsize=7)
                if col == 0:
                    axes[row, col].set_ylabel(f"head {row}", fontsize=7)

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        result_images.append(Image.open(buf).copy())
        plt.close(fig)

    return result_images

def visualize_matrix(t):
    t = t.squeeze(0)  # [4, 65, 65]
    num_heads = t.shape[0]
    fig, axes = plt.subplots(1, num_heads, figsize=(num_heads * 3, 3))
    for i in range(num_heads):
        axes[i].imshow(t[i].cpu().detach().numpy(), cmap='viridis')
        axes[i].set_title(f'Head {i}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()