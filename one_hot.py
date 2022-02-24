import torch
import common


def text2Vec(text):
    vec = torch.zeros(common.captcha_size, len(common.captcha_array))
    # print(vec)
    for i in range(len(text)):
        # print(common.captcha_array.index(text[i]))
        vec[i, common.captcha_array.index(text[i])] = 1
    return vec


def vec2Text(vec):
    vec = torch.argmax(vec, dim=1)  # 把为1的取出来
    # print(vec)
    text = ''
    for i in vec:
        text += common.captcha_array[i]
    return text


if __name__ == '__main__':
    vec = text2Vec("0×4=？")

    print(vec, vec.shape)
    print(vec2Text(vec))
