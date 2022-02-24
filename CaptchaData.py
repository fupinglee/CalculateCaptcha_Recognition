import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# 加载所有图片，并将验证码向量化
from one_hot import text2Vec


def make_dataset(data_path):
    img_names = os.listdir(data_path)
    samples = []
    for img_name in img_names:
        img_path = data_path + img_name
        target_str = img_name.split('_')[0].lower()
        samples.append((img_path, target_str))
    return samples


class CaptchaData(Dataset):
    def __init__(self, data_path, transform=None):
        super(Dataset, self).__init__()
        self.transform = transform
        self.samples = make_dataset(data_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, target = self.samples[index]
        target = text2Vec(target)
        target = target.view(1, -1)[0]
        img = Image.open(img_path)
        img = img.resize((160, 60))
        img = img.convert('RGB')  # img转成向量
        if self.transform is not None:
            img = self.transform(img)
        return img, target


if __name__ == '__main__':
    train_dataset = CaptchaData('./datasets/test/')
    print(len(train_dataset))
    img, target = train_dataset[0]
    print(np.asarray(img))
    img.show()
    print(np.asarray(img).shape)
    # print(vec2Text(target.view( -1)))
