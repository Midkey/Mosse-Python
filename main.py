import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

IMAGE_HEIGHT = 360
IMAGE_WIDTH = 480


# gt_center = [194, 306]
# gt_center = [131, 196]


def gaussian_2d(center):
    height, width = np.shape(g)
    for i in range(height):
        for j in range(width):
            dis = (i - center[0]) ** 2 + (j - center[1]) ** 2
            g[i, j] = np.exp(-dis / (200))


eta = 0.125
data_path = 'surfer'
image_list = os.listdir(data_path)

f = cv2.imread(os.path.join(data_path, image_list[0]))
init_box = cv2.selectROI('123', f)
left, top = init_box[0], init_box[1]
center = init_box[0] + init_box[2] / 2, init_box[1] + init_box[3] / 2
width = init_box[2]
height = init_box[3]
crop_img = cv2.cvtColor(f[int(center[1] - height / 2):int(center[1] + height / 2),
                          int(center[0] - width / 2):int(center[0] + width / 2)], cv2.COLOR_RGB2GRAY)
# cv2.imshow('12', crop_img)
# cv2.waitKey(0)
F = np.fft.fft2(crop_img)
g = np.zeros((init_box[3], init_box[2]))

gaussian_2d((init_box[3] / 2, init_box[2] / 2,))

G = np.matrix(np.fft.fft2(g))

A = np.multiply(G, F.conjugate())
B = np.multiply(F, F.conjugate())
# G = np.log(np.abs(G))


for img_path in image_list[1:]:
    f = cv2.imread(os.path.join(data_path, img_path))
    crop_img = cv2.cvtColor(f[int(center[1] - height / 2):int(center[1] + height / 2),
                              int(center[0] - width / 2):int(center[0] + width / 2)], cv2.COLOR_RGB2GRAY)
    cv2.resize(crop_img, (width, height), crop_img)
    F = np.fft.fft2(crop_img)
    # F = np.log(np.abs(F))
    H = (A / B).conjugate()
    G = np.multiply(F, H.conjugate())
    g = np.fft.ifft2(G)
    cv2.imshow('g', np.asarray(g, dtype=np.uint8))
    cv2.waitKey(10)
    m, n = np.shape(g)
    index = int(g.argmax())
    x = int(index / n)
    y = index % n
    center = left + y, top + x
    left, top = int(center[0] - width/2), int(center[1] - height/2)
    # cv2.circle(f, (left, top), 10, (255, 0, 0))
    f = cv2.rectangle(f, (left, top), (left + width, top + height), (0, 0, 255), 2)
    cv2.imshow('12', f)
    cv2.waitKey(10)
    A = eta * np.multiply(G, F.conjugate()) + (1 - eta) * A
    B = eta * np.multiply(F, F.conjugate()) + (1 - eta) * B
