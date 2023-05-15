import sys

sys.path.insert(0, '..')
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
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import confusion_matrix, classification_report

cls = {"Anger": 0, "Disgust": 1, "Fear": 2, "Happiness": 3, "Neutral": 4, "Sadness": 5, "Surprise": 6}


def draw_confusion_matrix(label_true, label_pred, label_name, normlize, title="Confusion Matrix", pdf_save_path=None,
                          dpi=100):

    cm = confusion_matrix(label_true, label_pred)
    if normlize:
        row_sums = np.sum(cm, axis=1)  # 计算每行的和
        cm = cm / row_sums[:, np.newaxis]  # 广播计算每个元素占比

    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)

    plt.tight_layout()

    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = float(format('%.2f' % cm[i, j]))
            plt.text(j, i, value, verticalalignment='center', horizontalalignment='center', color=color)

    plt.show()
    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)


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
        torch.save(model.state_dict(), 'resnet50_MODEL.pth')


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


def gradcam_for_each_class(model, data_path):
    model.eval()
    for i in cls.keys():
        img_path = data_path + '/' + i + '/' + os.listdir(data_path + '/' + i)[0]
        img = Image.open(img_path)
        img = img.resize((224, 224))
        img = np.float32(img) / 255
        input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        output = model(input_tensor)
        print(F.softmax(output, dim=1).data)
        print(output.argmax(dim=1).data)

        targets = [ClassifierOutputTarget(cls[i])]
        target_layers = [model.conv5_3_relu]
        with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
            grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
            cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
            ax = plt.imshow(cam_image)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.savefig(f'{i}_CAM.jpg')


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


def visualize():
    model = resnet50_face_sfew_dag("wangqingyu/resnet50_face_sfew_dag.pth", num_classes=7)
    model.load_state_dict(torch.load('resnet50_MODEL.pth'))
    gradcam_for_each_class(model, "rafdb/rafdb/val")


def draw_pic():

    model = resnet50_face_sfew_dag("wangqingyu/resnet50_face_sfew_dag.pth", num_classes=7)
    model.load_state_dict(torch.load('resnet50_MODEL.pth'))
    data_loader = load_data("rafdb/rafdb/val")
    model.cuda()
    model.eval()
    labels_name = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

    y_gt = []
    y_pred = []
    with torch.no_grad():
        for batch_idx, (input, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
            input = input.cuda()
            labels_pd = model(input)
            predict_np = np.argmax(labels_pd.cpu().detach().numpy(), axis=-1)
            labels_np = target.numpy()

            y_pred.extend(predict_np)
            y_gt.extend(labels_np)

    print(classification_report(y_gt, y_pred))

    draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=labels_name,
                          normlize=True,
                          title="Confusion Matrix on rafdb",
                          pdf_save_path="Confusion_Matrix_on_rafdb.jpg",
                          dpi=300)

def draw_num_each_class():
    dataloader = load_data("rafdb/rafdb/train")
    labels = []
    for batch_idx, (input, target) in tqdm(enumerate(dataloader), total=len(dataloader)):
        labels.extend(target.numpy())

    labels = np.array(labels)
    # draw
    labels_name = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
    plt.figure(figsize=(10, 5))
    plt.bar(labels_name, np.bincount(labels), width=0.5)
    plt.xlabel("Class")
    plt.ylabel("Number")
    plt.title("Number of each class on rafdb")
    plt.show()
    plt.savefig("Number_of_each_class_on_rafdb.jpg", dpi=300)


if __name__ == '__main__':
    Epochs = 10
    # main()
    # visualize()
    draw_pic()
