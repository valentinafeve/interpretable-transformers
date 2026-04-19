from torchvision import transforms
import torchvision
import torch
from matplotlib import pyplot as plt


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

    def get_samples_from_indices(self, indices, device="mps"):
        if self.testset is None:
            raise ValueError("Dataset not loaded. Call get_loaders() first.")
        images = torch.stack([self.testset[i][0] for i in indices]).to(device)
        labels = torch.tensor([self.testset[i][1] for i in indices]).to(device)
        return images, labels

    def visualize_sample(self, indices=None, return_images=False):
        """
        Visualizes a batch of random images if the indices are not provided, otherwise visualizes the images corresponding to the provided indices.
        Creates a grid of images with their corresponding labels as titles.
        """
        if self.testset is None:
            raise ValueError("Dataset not loaded. Call get_loaders() first.")

        if indices is None:
            indices = torch.randperm(len(self.testset))[:16].tolist()

        images, labels = self.get_samples_from_indices(indices, device="cpu")
        labels = labels.tolist()
        images = images * 0.5 + 0.5  # unnormalize

        n = len(indices)
        ncols = min(8, n)
        nrows = (n + ncols - 1) // ncols

        _, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
        axes = [axes] if n == 1 else axes.flatten() if nrows > 1 else list(axes)

        for ax, img, label in zip(axes, images, labels):
            ax.imshow(img.permute(1, 2, 0).squeeze(), cmap='gray' if self.n_channels == 1 else None)
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
            transforms.RandomHorizontalFlip(p=0.5),
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
