import cv2 as cv
import numpy as np

fname = 'D:/UoM/project/dataset/1750m_positive/1.jpg'
img = cv.imread(fname)
# 画矩形框
cv.rectangle(img, (828, 552), (852, 575), (0, 255, 0), 2)
# 标注文本
cv.imshow('001_new.jpg', img)
cv.waitKey(0)
cv.destroyAllWindows()
