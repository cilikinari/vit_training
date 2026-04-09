import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

def get_dataloader(batch_size):

    # transform
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

    # dataset
    train_dataset = datasets.ImageFolder(
        root='./kaggle/train',
        transform=transform
    )

    val_dataset = datasets.ImageFolder(
        root='./kaggle/val',
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
        shuffle=False,  
        batch_size=batch_size
    )

    return train_loader, val_loader