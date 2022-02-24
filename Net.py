import torch
from torch import nn
import common

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 第一层神经网络
        # nn.Sequential: 将里面的模块依次加入到神经网络中
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 3通道变成16通道，图片：60*160
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 图片：22*70
        )
        # 第2层神经网络
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),  # 16通道变成64通道，图片：20*68
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 图片：10*34
        )
        # 第3层神经网络
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),  # 16通道变成64通道，图片：8*32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 图片：4*16
        )
        # 第4层神经网络
        self.fc1 = nn.Sequential(
            nn.Linear(13824, 1024),
            nn.Dropout(0.2),  # drop 20% of the neuron
            nn.ReLU()
        )
        # 第5层神经网络
        self.fc2 = nn.Linear(1024, common.captcha_size * common.captcha_array.__len__())  # 6:验证码的长度， 37: 字母列表的长度

    # 前向传播
    def forward(self, x):
        x = x.to(device)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    data = torch.ones(1, 3, 60, 160)  # 64张图片 1表示灰色
    m = Net()
    x = m(data)
    print(x.shape)
