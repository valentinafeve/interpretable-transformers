"""
clothing_dataset.py
-------------------
Descarga el dataset `valentinafevu/clothing_described` desde HuggingFace y lo
envuelve en una clase compatible con torchvision (mismo formato que CIFAR10).

Cada ítem retorna: (PIL.Image, int_label)  ← igual que torchvision.datasets.CIFAR10

Uso rápido:
    from clothing_dataset import prepare_clothing_data, ClothingDataset
    trainloader, testloader, classes = prepare_clothing_data(batch_size=32)
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from datasets import load_dataset


# ---------------------------------------------------------------------------
# 1. Custom Dataset (compatible con torchvision)
# ---------------------------------------------------------------------------

class ClothingDataset(Dataset):
    """
    Wrapper sobre el HuggingFace dataset 'valentinafevu/clothing_described'
    con la misma interfaz que torchvision.datasets.CIFAR10.

    dataset[i] → (PIL.Image, int_label)

    Si se pasa un transform, dataset[i] → (Tensor, int_label)
    """

    # Nombre del dataset en HuggingFace
    HF_DATASET_NAME = "valentinafevu/clothing_described"

    IMAGE_COLUMN = "image"
    TEXT_COLUMN  = "text"   # la columna de descripción (ej. "a blue floral dress")

    def __init__(self, split="train", transform=None, test_ratio=0.2, seed=42):
        """
        Args:
            split      : "train" o "test". El dataset original solo tiene un split
                         'train', así que lo dividimos internamente (80/20 por defecto).
            transform  : torchvision transform a aplicar a cada imagen
            test_ratio : fracción del dataset que va al split de test (default 0.2)
            seed       : semilla para reproducibilidad

        La clase se extrae de la ÚLTIMA PALABRA del campo 'text'.
        Ej: "a casual cotton dress" → clase = "dress"
        """
        print(f"[ClothingDataset] Descargando '{self.HF_DATASET_NAME}'...")
        full = load_dataset(self.HF_DATASET_NAME, split="train")

        # Dividir en train / test reproduciblemente
        split_dataset = full.train_test_split(test_size=test_ratio, seed=seed)
        self.hf_dataset = split_dataset[split]  # "train" o "test"
        print(f"[ClothingDataset] Split '{split}': {len(self.hf_dataset)} muestras")
        self.transform = transform

        # -- Extraer clases a partir de la última palabra del campo text --
        last_words = [row[self.TEXT_COLUMN].strip().split()[-1].lower()
                      for row in self.hf_dataset]
        self.classes = sorted(set(last_words))          # lista ordenada de categorías únicas
        self._class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self._labels = [self._class_to_idx[w] for w in last_words]  # cache de labels

        print(f"[ClothingDataset] {len(self)} muestras | {len(self.classes)} clases: {self.classes}")

    # -- Interfaz de Dataset --

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]

        # Imagen
        image = item[self.IMAGE_COLUMN]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")  # garantiza 3 canales

        # Etiqueta: entero (índice de la clase)
        label = self._labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# ---------------------------------------------------------------------------
# 2. Función prepare_clothing_data  (reemplaza a prepare_data de CIFAR10)
# ---------------------------------------------------------------------------

IMAGE_SIZE = 32  # tamaño al que se redimensionan todas las imágenes


def prepare_clothing_data(
    batch_size=4,
    num_workers=2,
    train_sample_size=None,
    test_sample_size=None,
    train_split="train",
    test_split="test",
    image_size=IMAGE_SIZE,
):
    """
    Descarga el dataset de ropa y devuelve DataLoaders con el mismo formato
    que prepare_data() usa con CIFAR10.

    Returns:
        trainloader, testloader, classes
    """

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(
            (image_size, image_size),
            scale=(0.8, 1.0),
            ratio=(0.75, 1.333),
            interpolation=2,
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = ClothingDataset(split=train_split, transform=train_transform)
    testset  = ClothingDataset(split=test_split,  transform=test_transform)

    # Sub-muestreo opcional
    if train_sample_size is not None:
        indices = torch.randperm(len(trainset))[:train_sample_size]
        trainset = Subset(trainset, indices)

    if test_sample_size is not None:
        indices = torch.randperm(len(testset))[:test_sample_size]
        testset = Subset(testset, indices)

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Obtener clases del dataset base (aunque esté envuelto en Subset)
    base_train = trainset.dataset if isinstance(trainset, Subset) else trainset
    classes = tuple(base_train.classes)

    return trainloader, testloader, classes


# ---------------------------------------------------------------------------
# 3. visualize_images  (versión para el dataset de ropa)
# ---------------------------------------------------------------------------

def visualize_images_clothing(n=30, split="train", image_size=IMAGE_SIZE):
    """
    Muestra una grilla de imágenes del dataset de ropa, igual que
    visualize_images() de CIFAR10.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    dataset = ClothingDataset(split=split, transform=None)
    classes = dataset.classes

    indices = torch.randperm(len(dataset))[:n]
    images = [np.asarray(dataset[i][0]) for i in indices]
    labels = [dataset[i][1] for i in indices]

    cols = 5
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(12, 2.5 * rows))
    for i in range(n):
        ax = fig.add_subplot(rows, cols, i + 1, xticks=[], yticks=[])
        ax.imshow(images[i])
        ax.set_title(classes[labels[i]], fontsize=8)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 4. Script de inspección: corre esto primero para ver la estructura
# ---------------------------------------------------------------------------

def inspect_dataset():
    """
    Imprime la estructura del dataset para que puedas verificar
    los nombres de columnas y las clases.
    """
    from datasets import load_dataset
    print("=" * 60)
    print(f"Inspeccionando: {ClothingDataset.HF_DATASET_NAME}")
    print("=" * 60)

    full = load_dataset(ClothingDataset.HF_DATASET_NAME, split="train")
    print(f"\nTotal de muestras: {len(full)}")
    print(f"Columnas: {full.column_names}")
    print(f"Features: {full.features}")

    # Mostrar ejemplo
    ejemplo_texto = full[0]["text"]
    clase_extraida = ejemplo_texto.strip().split()[-1].lower()
    print(f"\nEjemplo texto : '{ejemplo_texto}'")
    print(f"→ clase extraída (última palabra): '{clase_extraida}'")

    # Clases únicas
    last_words = sorted(set(row["text"].strip().split()[-1].lower() for row in full))
    print(f"\nClases únicas ({len(last_words)}): {last_words}")
    print(f"\nDivisión automática: {int(len(full)*0.8)} train / {int(len(full)*0.2)} test")


# ---------------------------------------------------------------------------
# 5. Actualización del config del ViT para el dataset de ropa
# ---------------------------------------------------------------------------

def get_vit_config_for_clothing(num_classes: int, image_size: int = IMAGE_SIZE):
    """
    Devuelve el config del ViT ajustado al número de clases del dataset
    de ropa. Úsalo en lugar del config hardcodeado de CIFAR10.

    Ejemplo:
        _, _, classes = prepare_clothing_data()
        config = get_vit_config_for_clothing(num_classes=len(classes))
    """
    return {
        "patch_size": 4,
        "hidden_size": 48,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "intermediate_size": 4 * 48,
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "initializer_range": 0.02,
        "image_size": image_size,
        "num_classes": num_classes,   # ← se ajusta automáticamente
        "num_channels": 3,
        "qkv_bias": True,
        "use_faster_attention": False,
    }


# ---------------------------------------------------------------------------
# Ejecución directa: python clothing_dataset.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Paso 1: inspeccionar estructura del dataset
    inspect_dataset()

    # Paso 2: crear los DataLoaders
    trainloader, testloader, classes = prepare_clothing_data(
        batch_size=4,
        train_sample_size=200,   # muestra pequeña para prueba rápida
        test_sample_size=50,
    )

    print(f"\nClases ({len(classes)}): {classes}")
    print(f"Batches de entrenamiento: {len(trainloader)}")
    print(f"Batches de test         : {len(testloader)}")

    # Verificar forma de un batch
    images, labels = next(iter(trainloader))
    print(f"\nForma del batch de imágenes: {images.shape}")  # [4, 3, 32, 32]
    print(f"Forma del batch de labels  : {labels.shape}")   # [4]
    print(f"Tipo de label[0]           : {type(labels[0].item())} = {labels[0].item()}")

    # Paso 3: visualizar algunas imágenes
    visualize_images_clothing(n=30)
