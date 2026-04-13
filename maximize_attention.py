"""
Attention Maximization
----------------------
Carga el modelo entrenado (pesos congelados) y optimiza una imagen aleatoria
para maximizar la atención de la cabeza 0 del último bloque transformer.

Target: attention weight máxima que el token CLS (fila 0) le presta
a cualquier patch token — en el último bloque, cabeza 0.
"""

import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import ViTForClassfication

BASE_DIR   = "experiments/vit-with-10-epochs"
CHECKPOINT = f"./vit_final_2026-04-08-21-58-16.pt"

with open(f"{BASE_DIR}/config.json") as f:
    config = json.load(f)

model = ViTForClassfication(config)
model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
model.eval()

for p in model.parameters():
    p.requires_grad_(False)

print("Modelo cargado y congelado.")
print(f"  Bloques: {config['num_hidden_layers']}")
print(f"  Cabezas: {config['num_attention_heads']}")
print(f"  Tokens:  {(config['image_size'] // config['patch_size'])**2 + 1} (patches + CLS)")


TARGET_BLOCK = config["num_hidden_layers"] - 1   # último bloque (índice 3)
TARGET_HEAD  = 0                                  # cabeza 0
TARGET_ROW   = 0                                  # fila 0 = token CLS

LR     = 0.05
STEPS  = 400
L2_REG = 1e-4   # suavizado suave para que la imagen no explote

torch.manual_seed(0)
image = nn.Parameter(torch.randn(1, 3, config["image_size"], config["image_size"]) * 0.1)
optimizer = torch.optim.Adam([image], lr=LR)

print(f"\nOptimizando imagen para maximizar atención:")
print(f"  Bloque {TARGET_BLOCK} | Cabeza {TARGET_HEAD} | Fila {TARGET_ROW} (CLS)")
print()

losses = []
for step in range(STEPS):
    optimizer.zero_grad()

    _, all_attn = model(image, output_attentions=True)

    # all_attn[block]: (1, n_heads, seq_len, seq_len)
    attn_row = all_attn[TARGET_BLOCK][0, TARGET_HEAD, TARGET_ROW, :]  # (seq_len,)

    # Maximizar concentración de la fila usando entropía negativa:
    # target = -H(attn_row) = sum(p * log(p))  ≤ 0
    # Minimizar -target = minimizar entropía = maximizar concentración
    # El gradiente fluye por todos los tokens, no solo el máximo
    target = (attn_row * attn_row.clamp(min=1e-10).log()).sum()

    loss = -target + L2_REG * image.pow(2).mean()
    loss.backward()
    optimizer.step()

    losses.append(target.item())
    if step % 50 == 0:
        print(f"  Step {step:3d} | max attn: {target.item():.4f} | loss: {loss.item():.4f}")

print(f"\nAtención máxima final: {losses[-1]:.4f}")


# ── Visualización ──────────────────────────────────────────────────────────────

with torch.no_grad():
    _, all_attn = model(image, output_attentions=True)
    final_attn = all_attn[TARGET_BLOCK][0, TARGET_HEAD, TARGET_ROW, 1:].numpy()  # excluir CLS→CLS

# Reshape atención a grilla de patches
n_patches_side = config["image_size"] // config["patch_size"]  # 8
attn_grid = final_attn.reshape(n_patches_side, n_patches_side)

# Imagen optimizada: desnormalizar para visualizar
img_vis = image.detach()[0].permute(1, 2, 0).numpy()
img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-8)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle(f"Attention Maximization — Bloque {TARGET_BLOCK}, Cabeza {TARGET_HEAD}, Fila CLS", fontsize=12)

axes[0].imshow(img_vis)
axes[0].set_title("Imagen optimizada")
axes[0].axis("off")

im = axes[1].imshow(attn_grid, cmap="hot", interpolation="nearest")
axes[1].set_title("Atención CLS → patches")
axes[1].axis("off")
plt.colorbar(im, ax=axes[1], fraction=0.046)

axes[2].plot(losses)
axes[2].set_xlabel("Step")
axes[2].set_ylabel("Max attention weight")
axes[2].set_title("Curva de optimización")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("attention_maximization.png", dpi=120, bbox_inches="tight")
print("\nGuardado: attention_maximization.png")
plt.show()
