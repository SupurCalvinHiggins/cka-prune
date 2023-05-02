import torch
from torch import nn
from torchvision import transforms, datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import matplotlib.pyplot as plt


def get_loader(batch_size):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0], std=[1]),
        transforms.Lambda(lambda x: torch.flatten(x)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0], std=[1]),
        transforms.Lambda(lambda x: torch.flatten(x)),
    ])

    trainset = datasets.MNIST(root="./data",
                                train=True,
                                download=True,
                                transform=transform_train)
    testset = datasets.MNIST(root="./data",
                                train=False,
                                download=True,
                                transform=transform_test)

    train_sampler = RandomSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=batch_size,
                             num_workers=4,
                             pin_memory=True)

    return train_loader, test_loader


class LeNet(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()
        self.fc0 = nn.Linear(28 * 28, 300)
        self.fc1 = nn.Linear(300, 100)
        self.fc2 = nn.Linear(100, 10)
        self.dropout = nn.Dropout1d(dropout_rate)
    
    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


plt.show()

model = LeNet(0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0012)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.025, momentum=True)
criterion = nn.CrossEntropyLoss()
epochs = 20
batch_size = 60
train_loader, test_loader = get_loader(batch_size)
# pixels = next(iter(train_loader))[0][0][0]
# plt.imshow(pixels, cmap='gray')
# plt.show()

for epoch in range(1, epochs + 1):
    train_epoch_loss = 0

    model.train()
    for i, (batch_x, batch_y) in enumerate(train_loader):
        optimizer.zero_grad()
        pred_y = model(batch_x)
        loss = criterion(pred_y, batch_y)
        loss.backward()
        optimizer.step()
        train_epoch_loss += loss.item() * batch_size

    train_epoch_loss /= len(train_loader.dataset)

    model.eval()
    correct = 0
    for i, (batch_x, batch_y) in enumerate(test_loader):
        pred_y = torch.argmax(model(batch_x), axis=-1)
        correct += torch.sum(torch.eq(pred_y, batch_y))
    
    val_acc = correct / len(test_loader.dataset)

    print(f"Epoch {epoch}/{epochs}: loss = {loss.item()}, val_acc = {val_acc}")


torch.save(model.state_dict(), "lenet.model")