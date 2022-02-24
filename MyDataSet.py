import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import one_hot


class MyDataset(Dataset):
    def __init__(self, root_dir):
        super(MyDataset, self).__init__()
        self.image_path = [os.path.join(root_dir, image_name) for image_name in os.listdir(root_dir)]
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((60, 160)),
                transforms.Grayscale()  # 灰色
            ]
        )

    def __len__(self):
        return self.image_path.__len__()

    def __getitem__(self, index):
        image_path = self.image_path[index]
        # print(image_path)
        image = self.transforms(Image.open(image_path))
        ll = image_path.split("/")[-1]
        ll = ll.split("_")[0]
        label_tensor = one_hot.text2Vec(ll)  # [5,16]
        label_tensor = label_tensor.view(1, -1)[0]  # [5*16]
        # print(label)

        return image, label_tensor


if __name__ == '__main__':
    test_data = MyDataset("./datasets/test")
    img, label = test_data[1]
    # plt.imshow(np.asarray(img))
    # print(one_hot.vec2Text(label.view(6, -1)))
    print(img.shape, label, label.shape)
    print(np.asarray(img).shape)
