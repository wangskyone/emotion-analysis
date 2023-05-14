import sys
sys.path.insert(0,'..')
import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from facenet_pytorch import InceptionResnetV1, MTCNN
from tqdm import tqdm
from wangqingyu.resnet50_face_sfew_dag import resnet50_face_sfew_dag

cls = {"Anger": 0, "Disgust": 1, "Fear": 2, "Happiness": 3, "Neutral": 4, "Sadness": 5, "Surprise": 6}


class rafdb(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.labels = []
        self.imgs = []
        for i in cls.keys():
            self.imgs += [data_path + '/' + i + '/' + j for j in os.listdir(data_path + '/' + i)]
            self.labels += [cls[i]] * len(os.listdir(data_path + '/' + i))

    def __getitem__(self, index):
        img = Image.open(self.imgs[index])
        img = img.resize((224, 224))
        img = np.array(img)
        img = img / 255
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.labels)


def load_data(data_path):
    dataset = rafdb(data_path)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    return data_loader


def train(train_data_loader, model, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (input, target) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):
        input, target = input.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(input)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # print('Epoch ', epoch, '   ', 'Batch ', batch_idx, 'Loss ', running_loss/input.shape[0])
    print('Loss ', running_loss, 'Epoch ', epoch)
    if epoch == Epochs - 1:
        torch.save(model.state_dict(), f'{args.model}_MODEL.pth')


def test(test_data_loader, model, epoch):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (input, target) in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):
            input, target = input.cuda(), target.cuda()
            outputs = model(input)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print('Accuracy on test set: %d %% [%d/%d]    epoch: %d' % (100 * correct / total, correct, total, epoch))
    return 100 * correct / total


def main():
    model = resnet50_face_sfew_dag("wangqingyu/resnet50_face_sfew_dag.pth", num_classes=7)

    train_data_loader = load_data("rafdb/rafdb/train")
    test_data_loader = load_data("rafdb/rafdb/val")
    model.cuda()

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))

    for epoch in range(Epochs):
        train(train_data_loader, model, criterion, optimizer, epoch)
        test(test_data_loader, model, epoch)


if __name__ == '__main__':
    Epochs = 10
    main()
