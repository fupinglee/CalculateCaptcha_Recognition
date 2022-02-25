import os

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

import common
from CaptchaData import CaptchaData
from Net import Net
from one_hot import vec2Text
from train import calculat_acc

net = Net()


def predict(inputs):
    net.eval()  # 测试模式
    with torch.no_grad():
        outputs = net(inputs)
        outputs = outputs.view(-1, len(common.captcha_array))  # 每16个就是一个字符
    return vec2Text(outputs)


# 批量预测
def test():
    testpath = 'datasets/test/'
    transform = transforms.Compose([transforms.ToTensor()])  # 不做数据增强和标准化了
    test_data = CaptchaData(testpath, transform=transform)
    test_data_loader = DataLoader(test_data, batch_size=64, num_workers=0, shuffle=True, drop_last=True)
    # 加载模型
    model_path = 'model.pth'
    if os.path.exists(model_path):
        print('开始加载模型')
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()  # 测试模式
    acc, i = 0, 0
    with torch.no_grad():
        for inputs, labels in test_data_loader:
            pre = predict(inputs)

            target = labels.view(-1, len(common.captcha_array))  # 每16个就是一个字符
            target = vec2Text(target)  # 验证码文本
            outputs = net(inputs)
            acc += calculat_acc(outputs, labels)
            i += 1
    print('测试集正确率: %.3f %%' % (acc / i))


# 单张预测
def test_pic(path):
    img = Image.open(path)
    trans = transforms.Compose([
        transforms.Resize((60, 160)),
        # transforms.Grayscale(),
        transforms.ToTensor()
    ])
    img_tensor = trans(img)
    img_tensor = img_tensor.reshape(1, 3, 60, 160)  # 1张图片 1 灰色

    model_path = 'model.pth'
    net.eval()
    net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])

    output = net(img_tensor)
    output = output.view(-1, common.captcha_array.__len__())
    output_text = vec2Text(output)
    return output_text


# 下载新图片预测
def test_net(url):
    filepath = 'code.jpg'
    import requests
    res = requests.get(url)
    with open(filepath, "wb") as f:
        f.write(res.content)
    print(test_pic(filepath))


if __name__ == '__main__':
    test()
    # test_net("http://demo.ruoyi.vip/captcha/captchaImage?type=math&s=0.39236748354325024")
    # print(test_pic("datasets/test/0+7=？_75ba9179485bcfa30dd00a09fb027231.jpg"))
