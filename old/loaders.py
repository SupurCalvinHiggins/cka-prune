import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def get_mnist_loaders(batch_size):
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0], std=[1]),
            transforms.Lambda(lambda x: torch.flatten(x)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0], std=[1]),
            transforms.Lambda(lambda x: torch.flatten(x)),
        ]
    )

    trainset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform_train,
    )
    testset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform_test,
    )

    train_sampler = RandomSampler(trainset)
    test_sampler = SequentialSampler(testset)

    train_loader = DataLoader(
        trainset,
        sampler=train_sampler,
        batch_size=batch_size,
        pin_memory=True,
    )
    test_loader = DataLoader(
        testset,
        sampler=test_sampler,
        batch_size=batch_size,
        pin_memory=True,
    )

    return train_loader, test_loader