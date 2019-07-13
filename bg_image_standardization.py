import cv2 as cv
import numpy as np
import tensorflow as tf


def bg_image_std(image_path, sum):
    img = cv.imread(image_path)
    src = img[0:360, 0:360, :]
    width = 60
    height = 60
    cut_srcs = []
    count = 0
    for i in range(6):
        for j in range(6):
            cut_srcs.append(src[i*height:i*height + height, j*width: j*width + width])
    for cut_src in cut_srcs:
        count += 1
        name = "D:/UoM/project/dataset/cut_bg_image/" + str(sum) + "_" + str(count) + ".jpg"
        cv.imwrite(name, cut_src)


f = open('D:/UoM/project/dataset/1750m_bg/bg.txt')
img_path = 'D:/UoM/project/dataset/1750m_bg/'
context = f.readline()
sum = 0
while context:
    sum += 1
    context = context.split()
    path = img_path + context[0]
    bg_image_std(path, sum)
    context = f.readline()

cv.destroyAllWindows()