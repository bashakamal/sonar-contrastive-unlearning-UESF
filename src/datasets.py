import os
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms


def get_transforms(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),  # [-1, 1]
    ])


def create_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    image_size: int = 224,
    num_workers: int = 4,
    train_split: float = 0.7,
    val_split: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Creates train/val/test dataloaders from an ImageFolder directory.
    Directory structure:
        data_dir/
            class_1/
            class_2/
            ...
    """
    transform = get_transforms(image_size=image_size)
    full_dataset = ImageFolder(data_dir, transform=transform)
    class_names = full_dataset.classes

    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print(f"✅ Loaded dataset from: {data_dir}")
    print(f"   Total images: {total_size}")
    print(f"   Train / Val / Test: {train_size} / {val_size} / {test_size}")
    print(f"   Classes: {class_names}")

    return train_loader, val_loader, test_loader, class_names


class TripletSonarDataset(Dataset):
    """
    Triplet dataset used for Targeted Contrastive Unlearning (TCU).

    Anchor: any class (ship, plane, mine, human, etc.)
    Positive: same class as anchor
    Negative: always 'seafloor' class (background bias)
    """
    def __init__(self, base_dataset: Dataset, class_names: List[str], seabed_class_name: str = "seafloor"):
        self.dataset = base_dataset  # typically train split of ImageFolder
        self.class_names = class_names

        self.class_to_indices = {}
        for idx, (_, label) in enumerate(self.dataset):
            self.class_to_indices.setdefault(label, []).append(idx)

        if seabed_class_name not in class_names:
            raise ValueError(f"'seafloor' class not found in classes: {class_names}")

        self.seabed_label = class_names.index(seabed_class_name)

        print("🔧 TripletSonarDataset initialized with class distribution:")
        for label, indices in self.class_to_indices.items():
            cname = self.class_names[label]
            print(f"   {cname}: {len(indices)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        import random

        anchor_img, anchor_label = self.dataset[index]

        # Positive (same class)
        positive_indices = self.class_to_indices[anchor_label]
        if len(positive_indices) == 1:
            positive_index = positive_indices[0]
        else:
            positive_index = random.choice(positive_indices)
            while positive_index == index:
                positive_index = random.choice(positive_indices)

        positive_img, _ = self.dataset[positive_index]

        # Negative (seafloor)
        negative_indices = self.class_to_indices[self.seabed_label]
        negative_index = random.choice(negative_indices)
        negative_img, _ = self.dataset[negative_index]

        return anchor_img, positive_img, negative_img, anchor_label
