import cv2 as cv
import numpy as np

#降噪
def blur_demo(image):
    dst = cv.blur(image, (5, 5))
    cv.imshow("blur_demo", dst)

#去除椒盐噪声
def median_blur_demo(image):
    dst = cv.medianBlur(image, 5)
    cv.imshow("median_blur_demo", dst)


def custom_blur_demo(image):
    # kernel = np.ones([5, 5], np.float32)/25
    kernel = np.array([[0, -1, 0], [0, 5, 0], [0, -1, 0]], np.float32)
    dst = cv.filter2D(image, -1, kernel=kernel)
    cv.imshow("custom_blur_demo", dst)


src = cv.imread("D:/UoM/project/dataset/2000m/2019-06-11_154547.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
custom_blur_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()