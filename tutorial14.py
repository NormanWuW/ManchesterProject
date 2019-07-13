import cv2 as cv
import numpy as np


def big_image_binary(image):
    print(image.shape)
    cw = 255
    ch = 255
    h, w = image.shape[:2]
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", gray)
    for row in range(0, h, ch):
        for col in range(0, w, cw):
            roi = gray[row:row+ch, col:cw+col]
            print(np.std(roi), np.mean(roi))
            dev = np.std(roi)
            if dev < 15:
                gray[row:row + ch, col:cw + col] = 255
            else:
                # ret, dst = cv.threshold(roi, 127, 255, cv.THRESH_BINARY | cv.THRESH_TRUNC)
                # dst = cv.adaptiveThreshold(roi, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 127, 20)
                ret, dst = cv.threshold(roi, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
                gray[row:row + ch, col:cw + col] = dst

    cv.imwrite("D:/result.png", gray)


src = cv.imread("D:/FIRST TIME CHICKEN.png")
# cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# cv.imshow("input image", src)
big_image_binary(src)
cv.waitKey(0)
cv.destroyAllWindows()