from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler


def get_imagenet_test_loader(batch_size):
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    testset = datasets.ImageFolder(
        root='/mnt/fastdata/datasets/ILSVRC2012-test-partial/', 
        transform=transform_test
    )
    test_sampler = RandomSampler(testset)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=batch_size,
                             num_workers=4,
                             pin_memory=True)

    return test_loader


def get_cifar10_val_loader(batch_size):
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    testset = datasets.CIFAR10(
        root='`pytorch-mixer/data/', 
        train=False,
        transform=transform_test,
        download=True,
    )
    test_sampler = RandomSampler(testset)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=batch_size,
                             num_workers=4,
                             pin_memory=True)

    return test_loader