import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import random
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import torch.utils.data as data
import torch.nn.functional as F
import dataset


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
        self.conv1 = nn.Conv2d(1, 6, (5,5))
        # Layer 2: Convolutional. Output = 10x10x16.
        self.conv2 = nn.Conv2d(6, 16, (5,5))
        # Layer 3: Fully Connected. Input = 400. Output = 120.
        self.fc1   = nn.Linear(400, 120)
        # Layer 4: Fully Connected. Input = 120. Output = 84.
        self.fc2   = nn.Linear(120, 84)
        # Layer 5: Fully Connected. Input = 84. Output = 10.
        self.fc3   = nn.Linear(84, 10)



    def forward(self, x):
        # Activation. # Pooling. Input = 28x28x6. Output = 14x14x6.
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
         # Activation. # Pooling. Input = 10x10x16. Output = 5x5x16.
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        # Flatten. Input = 5x5x16. Output = 400.
        x = x.flatten(start_dim=1)
        # Activation.
        x = F.relu(self.fc1(x))
        # Activation.
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def train(epoch):
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = net(images.to(device))
        loss = criterion(output, labels.to(device))

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i+1)

        # if i % 10 == 0:
        #    print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

        loss.backward()
        optimizer.step()


def evaluate(target_loader, target_dataset):
    predictions = []
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(target_loader):
        output = net(images.to(device))
        avg_loss += criterion(output, labels.to(device)).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred).to(device)).sum()
        predictions.append(pred)

    avg_loss /= len(target_dataset)
    #print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))
    accuracy    = float(total_correct) / len(target_dataset)
    return accuracy, np.array(torch.cat(predictions).cpu())
    #or if you are in latest Pytorch world
    #return accuracy, np.array(torch.vstack(predictions))


if __name__ == "__main__":
    EPOCHS = 75
    gpu = 3
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else 'cpu')
    train_kwargs = {'batch_size': 128, 'shuffle': True}
    valid_kwargs = {'batch_size': 128, 'shuffle': False}
    test_kwargs = {'batch_size': 128, 'shuffle': False}

    transform = transforms.Compose([
        # Pad images with 0s
        transforms.Pad((0, 4, 4, 0), fill=0, padding_mode='constant'),

        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset_full = datasets.MNIST('../data', train=True, download=True,
                                  transform=transform)
    valid_size = 5000
    train_size = 20000
    remain = len(dataset_full) - valid_size - train_size
    dataset_train, dataset_valid, _ = torch.utils.data.random_split(dataset_full, [train_size, valid_size, remain])

    dataset_test = datasets.MNIST('../data', train=False,
                                  transform=transform)
    # label_path = "/tmp/pycharm_project_648/coursework2/src/CGAN/labels.npy"
    # imgs_path = "/tmp/pycharm_project_648/coursework2/src/CGAN/gen_imgs/"
    # labels = np.load(label_path)
    # cgan_data = dataset.CGANDataSet(imgs_path, labels)
    # dataset_train = torch.utils.data.ConcatDataset([dataset_train, cgan_data])
    train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, **valid_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    net = LeNet().to(device)
    print(net)

    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    print("Training...")
    print()
    for e in range(1, EPOCHS):
        train(e)
        validation_accuracy, validation_predictions = evaluate(valid_loader, dataset_valid)
        print("EPOCH {} ...".format(e))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    torch.save(
        {
            'lenet': net.state_dict(),
            'opt': optimizer.state_dict(),
        },
        ('model_real.model'),
    )
    print("Model saved")

    test_accuracy, test_predictions = evaluate(test_loader, dataset_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
