import cv2 as cv
import numpy as np


def fill_color_demo(image):
    copyImg = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros([h+2, w+2], np.uint8)
    cv.floodFill(copyImg, mask, (30, 30), (0, 255, 255), (100, 100, 100), (50, 50, 50), cv.FLOODFILL_FIXED_RANGE)
    cv.imshow("fill_color_demo", copyImg)


def fill_binary(image):
    image = np.zeros([400, 400, 3], np.uint8)
    image[100:300, 100:300, :] = 255
    # cv.floodFill(copyImg, mask, (30, 30), (0, 255, 255), (100, 100, 100), (50, 50, 50), cv.FLOODFILL_FIXED_RANGE)
    cv.imshow("fill_color_demo", image)

    mask = np.ones([402, 402, 1], np.uint8)
    mask[101:301, 101:301] = 0
    cv.floodFill(image, mask, (200, 200), (0, 0, 255), cv.FLOODFILL_MASK_ONLY)
    cv.imshow("fillled binary", image)


src = cv.imread("D:/FIRST TIME CHICKEN.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

# face = src[50:120, 80:380]
# gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
# backface = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
# src[50:120, 80:380] = backface
# print("-----channel number is different------")
# print(backface.shape)
# print(gray.shape)
# cv.imshow("gray", gray)
# cv.imshow("backface", backface)
fill_binary(src)
cv.waitKey(0)

cv.destroyAllWindows()
