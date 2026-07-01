"""Shared transform definitions and eight-view TTA pipelines."""

from torchvision import transforms
from torchvision.transforms import RandAugment

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def train_transform(img_size: int = IMG_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05),
        RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.2),
    ])


def val_transform(img_size: int = IMG_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def tta_transforms(img_size: int = IMG_SIZE) -> list:
    """
    Returns a list of 8 deterministic transforms for Test-Time Augmentation.
    Average predictions across all transforms for better robustness.
    """
    base = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
    return [
        transforms.Compose(base),
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]),
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]),
        transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]),
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation((90, 90)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]),
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation((180, 180)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]),
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation((270, 270)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]),
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]),
    ]
