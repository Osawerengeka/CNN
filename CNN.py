
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
import torchvision
import torch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=2)
        self.batch1 = nn.BatchNorm2d(64)
        self.batch2 = nn.BatchNorm2d(128)
        self.batch3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=2)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=2)
        self.conv6 = nn.Conv2d(256, 256, 3)
        self.fc1 = nn.Linear(7 * 7 * 256, 512)
        self.fc2 = nn.Linear(512, 10)
        self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.batch1(F.relu(self.conv1(x)))
        x = self.batch1(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = self.batch2(F.relu(self.conv3(x)))
        x = self.batch2(F.relu(self.conv4(x)))
        x = self.pool(x)
        x = self.batch3(F.relu(self.conv5(x)))
        x = self.batch3(F.relu(self.conv6(x)))
        x = x.view(-1, 7 * 7 * 256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return F.softmax(x, dim=1)


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = datasets.MNIST(root='./mnist', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./mnist', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=0)

dev = "cpu"
device = torch.device(dev)


plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False


def imshow(img, xticks=[], yticks=[], labels=[], xlabel=None, ylabel=None):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.figure(figsize=(15, 15))
    plt.xticks(ticks=xticks, labels=labels)
    plt.yticks(ticks=yticks, labels=labels)
    if xlabel:
        plt.xlabel('True class', fontsize=25)
        plt.ylabel('Predicted class', fontsize=25)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images[:8]))

def train_net(net, trainloader, testloader, critertion, optimizer, epoch_n):
    for epoch in range(5):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i * 64) % (2048 * 4) == 0 and i != 0:
                print(f'Эпоха: {epoch + 1}, объектов обработано: {i * 64}, ошибка: {running_loss / 128}')
                running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in trainloader:
                images, labels = data
                outputs = net(images.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()

        print(f'Точность train: {100 * correct / total} %')

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()

        print(f'Точность test: {100 * correct / total} %')

    print('Обучение завершено')


net = Net()
net = net.to(device)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.Adadelta(net.parameters(), lr=learning_rate)
train_net(net, trainloader, testloader, criterion, optimizer, 4)

