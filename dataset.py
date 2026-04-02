import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_dataloader(batch_size):

    # transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    # dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    val_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # dataloader
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,  # ⚠️ jangan shuffle validation
        batch_size=batch_size
    )

    return train_loader, val_loader