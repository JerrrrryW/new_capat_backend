import torch
from torch import nn, functional, optim
import os
import torchvision
from torchvision import models, datasets, transforms
from PIL import Image

import torch.utils.data as tud
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import warnings
import matplotlib.pyplot as plt

n_classes = 17
epoches = 100
pretrain = False
device = torch.device("mps")


traindataset = datasets.ImageFolder(root='./SealData/train/', transform=transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
 
]))
 
testdataset = datasets.ImageFolder(root='./SealData/test/', transform=transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
 
]))


classes = testdataset.classes
print(classes)
 
model = models.resnet18(pretrained=pretrain)
if pretrain == True:
    for param in model.parameters():
        param.requires_grad = False
model.fc = nn.Linear(in_features=512, out_features=n_classes, bias=True)
model = model.to(device)


 
def train_model(model, train_loader, loss_fn, optimizer, epoch):
    model.train()
    total_loss = 0.
    total_corrects = 0.
    total = 0.
    for idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        preds = outputs.argmax(dim=1)
        total_corrects += torch.sum(preds.eq(labels))
        total_loss += loss.item() * inputs.size(0)
        total += labels.size(0)
    total_loss = total_loss / total
    acc = 100 * total_corrects / total
    print("轮次:%4d|训练集损失:%.5f|训练集准确率:%6.2f%%" % (epoch + 1, total_loss, acc))
    return total_loss, acc
 

 
 
def test_model(model, test_loader, loss_fn, optimizer, epoch):
    model.train()
    total_loss = 0.
    total_corrects = 0.
    total = 0.
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            total_loss += loss.item() * inputs.size(0)
            total_corrects += torch.sum(preds.eq(labels))
 
        loss = total_loss / total
        accuracy = 100 * total_corrects / total
        print("轮次:%4d|训练集损失:%.5f|训练集准确率:%6.2f%%" % (epoch + 1, loss, accuracy))
        return loss, accuracy
 
 
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
train_loader = DataLoader(traindataset, batch_size=32, shuffle=True)
test_loader = DataLoader(testdataset, batch_size=32, shuffle=True)
for epoch in range(0, epoches):
    loss1, acc1 = train_model(model, train_loader, loss_fn, optimizer, epoch)
    loss2, acc2 = test_model(model, test_loader, loss_fn, optimizer, epoch)
 
classes = testdataset.classes
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
 
# path = r'C:\Users\Administrator\Desktop\动物\dataset\test\leopard\img_test_850.jpg'  # 测试图片路径
# model.eval()
# img = Image.open(path)
# img_p = transform(img).unsqueeze(0).to(device)
# output = model(img_p)
# pred = output.argmax(dim=1).item()
# plt.imshow(img)
# plt.show()
# p = 100 * nn.Softmax(dim=1)(output).detach().cpu().numpy()[0]
# print('该图像预测类别为:', classes[pred])
 
# # 三分类
# print('类别{}的概率为{:.2f}%，类别{}的概率为{:.2f}%，类别{}的概率为{:.2f}%'.format(classes[0], p[0], classes[1], p[1], classes[2], p[2]))


# net = nn.Sequential(
#     # 这里使用一个11*11的更大窗口来捕捉对象。
#     # 同时，步幅为4，以减少输出的高度和宽度。
#     # 另外，输出通道的数目远大于LeNet
#     nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
#     nn.MaxPool2d(kernel_size=3, stride=2),
#     # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
#     nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
#     nn.MaxPool2d(kernel_size=3, stride=2),
#     # 使用三个连续的卷积层和较小的卷积窗口。
#     # 除了最后的卷积层，输出通道的数量进一步增加。
#     # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
#     nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
#     nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
#     nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
#     nn.MaxPool2d(kernel_size=3, stride=2),
#     nn.Flatten(),
#     # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
#     nn.Linear(6400, 4096), nn.ReLU(),
#     nn.Dropout(p=0.5),
#     nn.Linear(4096, 4096), nn.ReLU(),
#     nn.Dropout(p=0.5),
#     # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
#     nn.Linear(4096, 10))

# X = torch.randn(1, 1, 224, 224)
# for layer in net:
#     X=layer(X)
#     print(layer.__class__.__name__,'output shape:\t',X.shape)

# batch_size = 128

