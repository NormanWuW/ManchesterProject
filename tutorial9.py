import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def plot_demo(image):
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show("histogram")


def image_hist(image):
    color = ("blue", "green", "red")
    for i, color in enumerate(color):
        # print(enumerate(color))
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show("histogram")


src = cv.imread("D:/UoM/project/dataset/3000m/1.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
plot_demo(src)
image_hist(src)
cv.waitKey(0)
cv.destroyAllWindows()