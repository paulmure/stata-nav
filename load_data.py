from torchvision import datasets, transforms
import torch

TRAIN_PATH = 'split_dataset/train'
VAL_PATH = 'split_dataset/val'


def get_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform


def get_augmented_transform():
    transform = transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomCrop(224),
        transforms.GaussianBlur(kernel_size=(7, 13), sigma=(6, 7)),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform


def get_data_loader():
    train_set = datasets.ImageFolder(TRAIN_PATH, transform=get_transform())
    val_set = datasets.ImageFolder(VAL_PATH, transform=get_transform())

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=32, shuffle=False)

    assert (train_set.class_to_idx == val_set.class_to_idx)

    return len(train_set.classes), train_loader, val_loader


def get_augmented_data_loader(augmented_transform):
    train_set = datasets.ImageFolder(TRAIN_PATH, transform=augmented_transform)
    val_set = datasets.ImageFolder(VAL_PATH, transform=get_transform())

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=32, shuffle=False)

    assert (train_set.class_to_idx == val_set.class_to_idx)

    return len(train_set.classes), train_loader, val_loader
