import cv2 as cv
import numpy as np
import tensorflow as tf


def image_std(path):
    img = cv.imread(path)
    std_img = tf.image.per_image_standardization(img)
    with tf.Session() as sess:
        result = sess.run(std_img)
        cv.imshow("result", result)
    cv.waitKey(10)
    return result


def img_resize(img):
    # img = cv.imread(path)
    res = cv.resize(img, (30, 30), interpolation=cv.INTER_CUBIC)
    # cv.imshow('iker', res)
    # cv.waitKey(100)
    return res


f = open('D:/UoM/project/dataset/counter_extract_img_ori/extract_img_ori_info.txt')
img_path = 'D:/UoM/project/dataset/counter_extract_img_ori/'
result_path = 'D:/UoM/project/dataset/counter_extract_img/'
context = f.readline()
sum = 0
while context:
    sum += 1
    context = context.split()
    path = img_path + context[0]
    alter_image = image_std(path)
    result = img_resize(alter_image)
    name = result_path + str(sum) + ".jpg"
    cv.imwrite(name, result)
    context = f.readline()

cv.destroyAllWindows()
