import numpy as np
from PIL import Image
import pickle
import os
import matplotlib.image as plimg

CHANNEL = 3
WIDTH = 32
HEIGHT = 32


def visualize_cifar10(num_classes, num_per_class, root_src, root_destination):
    data = []
    labels = []
    classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    for i in range(num_classes):
        with open(root_src+"cifar-10-batches-py/data_batch_" + str(i + 1), mode='rb') as file:
            # 数据集在当脚本前文件夹下
            data_dict = pickle.load(file, encoding='bytes')
            data += list(data_dict[b'data'])
            labels += list(data_dict[b'labels'])

    img = np.reshape(data, [-1, CHANNEL, WIDTH, HEIGHT])

    for i in range(num_per_class):
        r = img[i][0]
        g = img[i][1]
        b = img[i][2]

        plimg.imsave(root_destination + 'pic4/' + str(i) + "r" + ".png", r)
        plimg.imsave(root_destination + 'pic4/' + str(i) + "g" + ".png", g)
        plimg.imsave(root_destination + 'pic4/' + str(i) + "b" + ".png", b)

        ir = Image.fromarray(r)
        ig = Image.fromarray(g)
        ib = Image.fromarray(b)
        rgb = Image.merge("RGB", (ir, ig, ib))

        name = "img-" + str(i) + "-" + classification[labels[i]] + ".png"
        rgb.save(root_destination + name, "PNG")

if __name__ == '__main__':
    root_scr = './datasets/CIFAR/'
    root_dest = './datasets/visualization_cifar10/'

    num_classes = 5
    n_per_class = 100
    visualize_cifar10(num_classes=num_classes, num_per_class=n_per_class,
                      root_src=root_scr,
                      root_destination=root_dest)

