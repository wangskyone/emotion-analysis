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
from resnet50_face_sfew_dag import resnet50_face_sfew_dag
import argparse
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=3)
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


class Resnet50(nn.Module):
    def __init__(self, num_classes=2):
        super(Resnet50, self).__init__()
        self.resnet50 = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
        self.resnet50.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.resnet50(x)
        return x


class GenKI_4K(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.img1 = os.listdir(data_path + '/1')
        self.img0 = os.listdir(data_path + '/0')
        self.imgs = self.img1 + self.img0
        self.labels = [1] * len(self.img1) + [0] * len(self.img0)
        if args.model == 'network':
            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

    def __getitem__(self, index):
        label = self.labels[index]
        img_path = self.data_path + '/' + str(label) + '/' + self.imgs[index]
        img = Image.open(img_path)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


def load_data(data_path):
    dataset = GenKI_4K(data_path)
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


def show_pic(data_loader):
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    print(images.shape)
    print(labels.shape)

    for i in range(8):
        ax = plt.subplot(2, 4, i + 1)
        ax.axis('off')
        plt.imshow(images[i].permute(1, 2, 0))
        plt.title(cls[labels[i].item()])

    plt.show()


def get_cam(model):
    model.eval()
    img = Image.open('datasets/test_folder/1/file1564.jpg')
    if args.model == 'network':
        img = img.resize((64, 64))
    else:
        img = img.resize((224, 224))
    img = np.float32(img) / 255
    input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    output = model(input_tensor)
    print(F.softmax(output, dim=1).data)

    targets = [ClassifierOutputTarget(1)]
    if args.model == 'network':
        target_layers = [model.conv2]
    elif args.model == 'inception':
        target_layers = [model.repeat_3[4].conv2d]
    elif args.model == 'resnet50':
        target_layers = [model.conv5_3_relu]
    with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
        grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
        cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
        ax = plt.imshow(cam_image)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        plt.title(f'{args.model}_CAM')
        plt.savefig(f'{args.model}_CAM.jpg')

    plt.show()


def main():
    # data path
    data = './datasets/'
    train_data_path = data + 'train_folder'
    test_data_path = data + 'test_folder'
    train_data_loader = load_data(train_data_path)
    test_data_loader = load_data(test_data_path)

    # show_pic(test_data_loader)

    model = get_model()
    print(f'model parameters: {cal_params(model)}M')
    model.cuda()

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))

    for epoch in range(Epochs):
        train(train_data_loader, model, criterion, optimizer, epoch)
        test(test_data_loader, model, epoch)


def visualize():
    model = get_model()
    model.load_state_dict(torch.load(f'{args.model}_MODEL.pth'))
    get_cam(model)


def get_model():
    if args.model == 'network':
        model = Network()
    elif args.model == 'inception':
        model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=2)
    elif args.model == 'resnet50':
        model = resnet50_face_sfew_dag("resnet50_face_sfew_dag.pth",num_classes=2)
    elif args.model == 'resnet50-imgnet':
        model = Resnet50()
    return model


def draw_pic():
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

    x = ['Easy CNN', 'InceptionResnetV1(Vggface2)', 'Resnet50(Vggface2)', 'Resnet50(ImageNet)']
    y = [84, 92, 93, 90]
    error = [1, 1, 0.95, 2]  # 误差或方差
    z = [2.01, 23.4, 23.5, 23.5]  # 第二个y轴的维度，此处为运行的时间

    fig = plt.figure(figsize=(14, 8), dpi=80)
    ax1 = fig.add_subplot(111)
    bar_width = 0.3  # 柱状图宽度
    ax1.bar(x, y, bar_width)  # 生成柱状图
    ax1.errorbar(x, y, yerr=error, capsize=3, elinewidth=2, fmt=' k,')  # 添加误差棒

    plt.ylim(0, 100)  # 限定左侧Y轴显示尺度范围
    plt.xlabel('模型', size=20)
    plt.ylabel('准确率', size=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.grid(linestyle="--", alpha=1, which='both')  # 添加网格线

    ax2 = ax1.twinx()  # 关键步骤，开始设置第二个Y轴的内容
    ax2.plot(x, z, label='y2', marker='o', color='red')
    plt.yticks(size=15)
    plt.ylim(0, 30)
    plt.ylabel('模型参数量(M)', size=20)
    plt.show()


def cal_params(model):
    params = 0
    for param in model.parameters():
        params += param.numel()
    return params / 1000000


if __name__ == '__main__':
    # torch.set_num_threads(1)
    Epochs = 10
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='network')
    args = parser.parse_args()
    cls = ['no smile', 'smile']
    # main()
    # visualize()
    draw_pic()
    print('Done......')
