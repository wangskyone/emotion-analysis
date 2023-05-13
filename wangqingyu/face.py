

import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size = 3)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size = 3)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc1 = torch.nn.Linear(3920, 512)
        self.fc2 = torch.nn.Linear(512, 2)
    def forward(self, x):
        # import pdb; pdb.set_trace()
        B, C, W, H = x.shape
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(B, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def load_data(data_path_yes, data_path_no):
    
    data = torch.empty(0, 3, 64, 64)
    label = torch.empty(0, 2)
    imgs_yes = os.listdir(data_path_yes)
    for img in imgs_yes:
        image = transforms.ToTensor()(np.array(plt.imread(data_path_yes + '/' + img))).contiguous()  # torch.Size([3, 64, 64])
        image_tensor = image.unsqueeze(0)
        data = torch.cat((data, image_tensor), 0)
        label = torch.cat((label, torch.Tensor([1, 0]).unsqueeze(0)), 0)  
    imgs_no = os.listdir(data_path_no)
    for img in imgs_no:
        image = transforms.ToTensor()(np.array(plt.imread(data_path_no + '/' + img))).contiguous()  # torch.Size([3, 64, 64])
        image_tensor = image.unsqueeze(0)
        data = torch.cat((data, image_tensor), 0)
        label = torch.cat((label, torch.Tensor([0, 1]).unsqueeze(0)), 0)
    # import pdb; pdb.set_trace()
    # label_hot = F.one_hot(label.long(), num_classes = 3).float()
    data = TensorDataset(data, label)  # torch.Size([32, 3, 64, 64])  torch.Size([32])
    data_loader = DataLoader(data, batch_size = 32, shuffle = True, num_workers = 2)

    return data_loader


def train(train_data_loader, model, criterion, optimizer, epoch):
    # import pdb; pdb.set_trace()
    running_loss = 0.0
    for batch_idx, (input, target) in enumerate(train_data_loader):
        input, target = input.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(input)
        # import pdb; pdb.set_trace()
        loss = criterion(output, target.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # print('Epoch ', epoch, '   ', 'Batch ', batch_idx, 'Loss ', running_loss/input.shape[0])
    print('Loss ', running_loss, 'Epoch ', epoch)
    

def test(test_data_loader, model, epoch):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(test_data_loader):
            # import pdb; pdb.set_trace()
            input, target = input.cuda(), target.cuda()
            outputs = model(input)
            _, predicted = torch.max(outputs.data, dim = 1)
            total += target.size(0)
            _, realone = torch.max(target.data, dim = 1)
            correct += (predicted == realone).sum().item()
    print('Accuracy on test set: %d %% [%d/%d]    epoch: %d' % (100 * correct / total, correct, total, epoch))
    return 100 * correct / total


def main():
    
    # data path
    data = '/home/wangqingyu/Face/smile-detection-master/datasets/'
    train_data_path_yes = data + 'train_folder/1'
    train_data_path_no = data + 'train_folder/0'
    test_data_path_yes = data + 'test_folder/1'
    test_data_path_no = data + 'test_folder/0'
    train_data_loader = load_data(train_data_path_yes, train_data_path_no)
    test_data_loader = load_data(test_data_path_yes, test_data_path_no)
    
    model = Network()
    model.cuda()
    
    Epochs = 100
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)
    
    for epoch in range(Epochs):
        train(train_data_loader, model, criterion, optimizer, epoch)
        test(test_data_loader, model, epoch)



if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
    print('Done......')
    
