import os

import pandas as pd
import torch
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms


class Dataset:
    def __init__(self):
        self.classes = ()
        self.n_channels = None
        self.n_classes = 0
        self.name = ""
        self.trainset = None
        self.testset = None
        self.train_transform = None
        self.test_transform = None
        self.multi_label = False

    def get_samples_from_indices(self, indices, device="mps", set="test"):
        dataset = self.testset if set == "test" else self.trainset
        if dataset is None:
            raise ValueError("Dataset not loaded. Call get_loaders() first.")
        images = torch.stack([dataset[i][0] for i in indices]).to(device)
        label_list = [dataset[i][1] for i in indices]
        if torch.is_tensor(label_list[0]):
            labels = torch.stack(label_list).to(device)
        else:
            labels = torch.tensor(label_list).to(device)
        return images, labels

    def visualize_sample(self, indices=None, return_images=False, set="test"):
        """
        Visualizes a batch of random images if the indices are not provided, otherwise visualizes the images corresponding to the provided indices.
        Creates a grid of images with their corresponding labels as titles.
        """
        if (self.testset is None and set == "test") or (self.trainset is None and set == "train"):
            raise ValueError("Dataset not loaded. Call get_loaders() first.")

        if indices is None:
            dataset = self.testset if set == "test" else self.trainset
            indices = torch.randperm(len(dataset))[:16].tolist()

        images, labels = self.get_samples_from_indices(indices, device="cpu", set=set)
        labels = labels.tolist()
        images = images * 0.5 + 0.5  # unnormalize

        n = len(indices)
        ncols = min(8, n)
        nrows = (n + ncols - 1) // ncols

        _, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
        axes = [axes] if n == 1 else axes.flatten() if nrows > 1 else list(axes)

        for ax, img, label in zip(axes, images, labels):
            ax.imshow(img.permute(1, 2, 0).squeeze(), cmap='gray' if self.n_channels == 1 else None)
            if self.multi_label:
                title = ", ".join(c for c, v in zip(self.classes, label) if v)
                ax.set_title(title, fontsize=6)
            else:
                ax.set_title(self.classes[label])
            ax.axis('off')

        for ax in axes[n:]:
            ax.axis('off')

        plt.tight_layout()
        plt.show()
        if return_images:
            return images, labels


class CIFAR10Dataset(Dataset):
    def __init__(self):
        super().__init__()
        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.n_channels = 3
        self.n_classes = len(self.classes)
        self.name = "CIFAR10"

        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.333), interpolation=2),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def get_loaders(self, batch_size=4, num_workers=2, train_sample_size=None, test_sample_size=None):
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=self.train_transform)
        self.trainset = trainset
        if train_sample_size is not None:
            indices = torch.randperm(len(trainset))[:train_sample_size]
            trainset = torch.utils.data.Subset(trainset, indices)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=self.test_transform)
        self.testset = testset
        if test_sample_size is not None:
            indices = torch.randperm(len(testset))[:test_sample_size]
            testset = torch.utils.data.Subset(testset, indices)

        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers)

        return trainloader, testloader, self.classes


class MNISTDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.classes = tuple(str(i) for i in range(10))
        self.n_channels = 1
        self.n_classes = len(self.classes)
        self.name = "MNIST"

        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad(padding=4, fill=0),
            transforms.RandomRotation(degrees=(-90, 90), fill=(0,)),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5, fill=0),
            transforms.RandomResizedCrop((28, 28), scale=(0.8, 1.0), ratio=(0.75, 1.333), interpolation=2),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def get_loaders(self, batch_size=4, num_workers=2, train_sample_size=None, test_sample_size=None):
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                              download=True, transform=self.train_transform)
        self.trainset = trainset
        if train_sample_size is not None:
            indices = torch.randperm(len(trainset))[:train_sample_size]
            trainset = torch.utils.data.Subset(trainset, indices)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)

        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                             download=True, transform=self.test_transform)
        self.testset = testset
        if test_sample_size is not None:
            indices = torch.randperm(len(testset))[:test_sample_size]
            testset = torch.utils.data.Subset(testset, indices)

        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers)

        return trainloader, testloader, self.classes


class _CelebACSVDataset(torch.utils.data.Dataset):
    """
    Reads the Kaggle export of CelebA (jessicali9530/celeba-dataset) instead of
    torchvision.datasets.CelebA, whose annotation files are only served from
    Google Drive and routinely hit its download quota.

    Expects `root` to directly contain list_attr_celeba.csv, list_eval_partition.csv
    and an img_align_celeba/ folder (optionally nested one level deeper, as Kaggle
    zips it: img_align_celeba/img_align_celeba/*.jpg).
    """

    _SPLIT_TO_PARTITION = {'train': 0, 'valid': 1, 'test': 2}

    def __init__(self, root, split='train', transform=None):
        self.transform = transform

        img_dir = os.path.join(root, 'img_align_celeba')
        if os.path.isdir(os.path.join(img_dir, 'img_align_celeba')):
            img_dir = os.path.join(img_dir, 'img_align_celeba')
        self.img_dir = img_dir

        attr_df = pd.read_csv(os.path.join(root, 'list_attr_celeba.csv'), index_col='image_id')
        self.attr_names = tuple(attr_df.columns)
        attr_df = (attr_df + 1) // 2  # map {-1, 1} -> {0, 1}

        if split != 'all':
            partition_df = pd.read_csv(os.path.join(root, 'list_eval_partition.csv'), index_col='image_id')
            partition = self._SPLIT_TO_PARTITION[split]
            keep_ids = partition_df.index[partition_df['partition'] == partition]
            attr_df = attr_df.loc[attr_df.index.intersection(keep_ids)]

        self.image_ids = attr_df.index.tolist()
        self.attr = torch.tensor(attr_df.values, dtype=torch.long)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.img_dir, self.image_ids[index])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.attr[index]



class CelebADataset(Dataset):
    """
    Multi-label attribute classification: each sample's label is a 40-dim
    binary vector (one entry per facial attribute), not a single class index.
    Loads from a Kaggle-style CelebA export (see _CelebACSVDataset) rather than
    torchvision.datasets.CelebA, since the latter's annotation files are only
    served from Google Drive and routinely hit its download quota.
    """

    def __init__(self, image_size=128, root='./data/celeba'):
        super().__init__()
        self.classes = (
            '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
            'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry',
            'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
            'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
            'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
            'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
            'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
            'Wearing_Necklace', 'Wearing_Necktie', 'Young',
        )
        self.n_channels = 3
        self.n_classes = len(self.classes)
        self.name = "CelebA"
        self.image_size = image_size
        self.multi_label = True
        self.root = root

        self.train_transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.test_transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def get_loaders(self, batch_size=4, num_workers=2, train_sample_size=None, test_sample_size=None):
        trainset = _CelebACSVDataset(root=self.root, split='train', transform=self.train_transform)
        self.trainset = trainset
        if train_sample_size is not None:
            indices = torch.randperm(len(trainset))[:train_sample_size]
            trainset = torch.utils.data.Subset(trainset, indices)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)

        testset = _CelebACSVDataset(root=self.root, split='test', transform=self.test_transform)
        self.testset = testset
        if test_sample_size is not None:
            indices = torch.randperm(len(testset))[:test_sample_size]
            testset = torch.utils.data.Subset(testset, indices)

        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers)

        return trainloader, testloader, self.classes


class OxfordIIITPetDataset(Dataset):
    def __init__(self, image_size=128):
        super().__init__()
        self.classes = (
            'Abyssinian', 'American Bulldog', 'American Pit Bull Terrier', 'Basset Hound',
            'Beagle', 'Bengal', 'Birman', 'Bombay', 'Boxer', 'British Shorthair',
            'Chihuahua', 'Egyptian Mau', 'English Cocker Spaniel', 'English Setter',
            'German Shorthaired', 'Great Pyrenees', 'Havanese', 'Japanese Chin', 'Keeshond',
            'Leonberger', 'Maine Coon', 'Miniature Pinscher', 'Newfoundland', 'Persian',
            'Pomeranian', 'Pug', 'Ragdoll', 'Russian Blue', 'Saint Bernard', 'Samoyed',
            'Scottish Terrier', 'Shiba Inu', 'Siamese', 'Sphynx', 'Staffordshire Bull Terrier',
            'Wheaten Terrier', 'Yorkshire Terrier',
        )
        self.n_channels = 3
        self.n_classes = len(self.classes)
        self.name = "OxfordIIITPet"
        self.image_size = image_size

        to_rgb = transforms.Lambda(lambda img: img.convert("RGB"))

        self.train_transform = transforms.Compose([
            to_rgb,
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.8, 1.0), ratio=(0.75, 1.333), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.test_transform = transforms.Compose([
            to_rgb,
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def get_loaders(self, batch_size=4, num_workers=2, train_sample_size=None, test_sample_size=None):
        trainset = torchvision.datasets.OxfordIIITPet(root='./data', split='trainval',
                                                       target_types='category', download=True,
                                                       transform=self.train_transform)
        self.trainset = trainset
        if train_sample_size is not None:
            indices = torch.randperm(len(trainset))[:train_sample_size]
            trainset = torch.utils.data.Subset(trainset, indices)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)

        testset = torchvision.datasets.OxfordIIITPet(root='./data', split='test',
                                                      target_types='category', download=True,
                                                      transform=self.test_transform)
        self.testset = testset
        if test_sample_size is not None:
            indices = torch.randperm(len(testset))[:test_sample_size]
            testset = torch.utils.data.Subset(testset, indices)

        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers)

        return trainloader, testloader, self.classes

