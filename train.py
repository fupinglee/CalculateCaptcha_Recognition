import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from Net import Net
import common
from CaptchaData import CaptchaData


def calculat_acc(output, target):
    output, target = output.view(-1,len(common.captcha_array)), target.view(-1,len(common.captcha_array))  # 每16个就是一个字符
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    target = torch.argmax(target, dim=1)
    output, target = output.view(-1, common.captcha_size), target.view(-1, common.captcha_size)  # 每5个字符是一个验证码
    target = target.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    c = 0

    for i, j in zip(target, output):
        if torch.equal(i, j):
            c += 1
    acc = c / output.size()[0] * 100
    return acc


def train(epoch_nums):
    net = Net()
    # 数据准备
    transform = transforms.Compose([transforms.ToTensor()])  # 不做数据增强和标准化了
    train_dataset = CaptchaData('./datasets/train/', transform=transform)
    train_data_loader = DataLoader(train_dataset, batch_size=32, num_workers=0, shuffle=True, drop_last=True)

    test_data = CaptchaData('./datasets/test/', transform=transform)
    test_data_loader = DataLoader(test_data, batch_size=128, num_workers=0, shuffle=True, drop_last=True)
    # 更换设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('当前设备是:', device)
    net.to(device)

    criterion = nn.MultiLabelSoftMarginLoss()  # 损失函数
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # 优化器

    # 加载模型
    model_path = 'model.pth'
    if os.path.exists(model_path):
        print('开始加载模型')
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 开始训练
    i = 1
    for epoch in range(epoch_nums):
        print("开始第{}批训练".format(epoch))
        running_loss = 0.0
        net.train()  # 神经网络开启训练模式
        for data in train_data_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # 数据发送到指定设备
            # 每次迭代都要把梯度置零
            optimizer.zero_grad()
            # 关键步骤
            # 前向传播
            outputs = net(inputs)
            # 计算误差
            loss = criterion(outputs, labels)
            # 后向传播
            loss.backward()
            # 优化参数
            optimizer.step()

            running_loss += loss.item()
            if i % 500 == 0:
                acc = calculat_acc(outputs, labels)
                print('第%s次训练正确率: %.3f %%, loss: %.3f' % (i, acc, running_loss / 2000))
                running_loss = 0
                # 保存模型
                torch.save({
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, model_path)
            i += 1
        # 结束一个epoch,计算测试集的正确率
        net.eval()  # 测试模式
        with torch.no_grad():
            for inputs, labels in test_data_loader:
                outputs = net(inputs)
                acc = calculat_acc(outputs, labels)
                print('测试集正确率: %.3f %%' % (acc))
                break  # 只测试一个batch

        # 每5个epoch 更新学习率
        if epoch % 5 == 4:
            for p in optimizer.param_groups:
                p['lr'] *= 0.9


if __name__ == '__main__':
    train(100)
