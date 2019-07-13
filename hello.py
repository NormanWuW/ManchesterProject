import cv2 as cv
import numpy as np

src = cv.imread("D:/FIRST TIME CHICKEN.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
cv.waitKey(0)
cv.destroyAllWindows()
print("hi python")