import torch
from torchvision import datasets, transforms
from .base_dataset import BaseDataset


class ImageClassificationDataset(BaseDataset):
    def __init__(self, root='./data/cifar10', train=True, transform=None):
        super().__init__()
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        self.dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def preprocess(self, data):
        # CIFAR-10 data is already preprocessed by torchvision, so we don't need to do anything here
        return data


def get_image_classification_dataloaders(batch_size=32, num_workers=2):
    train_dataset = ImageClassificationDataset(train=True)
    val_dataset = ImageClassificationDataset(train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
