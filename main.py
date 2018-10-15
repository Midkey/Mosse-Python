import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

IMAGE_HEIGHT = 360
IMAGE_WIDTH = 480
gt_center = [194, 306]
# gt_center = [131, 196]


def gaussian_2d(center):
    r = np.sqrt(center[0] ** 2 + center[1] ** 2)
    gt = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
    for i in range(IMAGE_HEIGHT):
        for j in range(IMAGE_WIDTH):
            dis = (i - center[0]) ** 2 + (j - center[1]) ** 2
            gt[i, j] = np.exp(-dis / (200))
    return gt

eta = 0.125
data_path = 'surfer'
image_list = os.listdir(data_path)
# img = cv2.imread(os.path.join(data_path, image_list[0]), 0)
# plt.imshow(img)
# plt.show()
H = None
for img_path in image_list:
    f = cv2.imread(os.path.join(data_path, img_path), 0)
    F = np.matrix(np.fft.fft2(f))
    # F = np.log(np.abs(F))
    if H is not None:
        G = np.multiply(F, H.conjugate())
        g = np.fft.ifft2(G)
        # cv2.imshow('123', g)
        # cv2.waitKey(0)
        m, n = np.shape(g)
        index = int(g.argmax())
        x = int(index/n)
        y = index % n
        img = cv2.circle(f, (y, x), 5, 255, 10)
        cv2.imshow('12', img)
        # g = gaussian_2d([x, y])
        # cv2.imshow('123', g)
        cv2.waitKey(100)
        # G = np.fft.fft2(g)
        H = eta * np.multiply(F, G.conjugate()) / np.multiply(F, F.conjugate()) + (1 - eta) * H
        H = np.matrix(H)
        # print()
    else:
        g = gaussian_2d(gt_center)
        cv2.imshow('11', g)
        cv2.waitKey(0)
        G = np.matrix(np.fft.fft2(g))
        # G = np.log(np.abs(G))
        H = np.multiply(F, G.conjugate()) / np.multiply(F, F.conjugate())
        H = np.matrix(H)
