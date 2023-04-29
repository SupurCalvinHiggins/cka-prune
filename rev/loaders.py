import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split


DATASET_FOLDER = "./data"


def build_transform():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0], std=[1]),
            transforms.Lambda(lambda x: torch.flatten(x)),
        ]
    )
    return transform


def build_datasets(transform):
    dataset = datasets.MNIST(
        root=DATASET_FOLDER, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=DATASET_FOLDER, train=False, download=True, transform=transform
    )
    return dataset, test_dataset


def split_dataset(dataset, split):
    left_dataset_size = int(len(dataset) * split)
    right_dataset_size = len(dataset) - left_dataset_size
    generator = torch.Generator().manual_seed(42)
    return random_split(dataset, [left_dataset_size, right_dataset_size], generator)


def build_samplers(train_dataset, val_dataset, test_dataset):
    train_sampler = RandomSampler(train_dataset)
    val_sampler = RandomSampler(val_dataset)
    test_sampler = SequentialSampler(test_dataset)
    return train_sampler, val_sampler, test_sampler


def get_loaders(batch_size, val_split = (1/12)):
    transform = build_transform()
    dataset, test_dataset = build_datasets(transform)
    train_dataset, val_dataset = split_dataset(dataset, val_split)
    train_sampler, val_sampler, test_sampler = build_samplers(
        train_dataset, val_dataset, test_dataset
    )
    
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=batch_size,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=batch_size,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader