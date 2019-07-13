import cv2 as cv
import numpy as np

def access_pixel(image):
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    print("width : %s, height: %s, channels : %s" % (width, height, channels))
    for row in range(height):
        for col in range(width):
            for c in range(channels):
                pv = image[row, col, c]
                image[row, col, c] = 255 - pv
    cv.imshow("pixels_demo", image)


def inverse(image):
    dst = cv.bitwise_not(image)
    cv.imshow("inverse demo", dst)


def create_image():
    # initialize
    # img = np.zeros([800, 800, 3], np.uint8)
    # img[:, :, 0] = np.ones([800, 800]) * 255
    # cv.imshow("new image", img)
    n1 = np.ones([3, 3], np.uint8)
    n1.fill(123)
    print(n1)

    n2 = n1.reshape([1, 9])
    print(n2)

    n3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], np.int32)
    n3.fill(9)
    print(n3)

    # img = np.zeros([800, 800, 1], np.uint8)
    # img[:, :, 0] = np.ones([800, 800]) * 127
    # cv.imshow("new image", img)


src = cv.imread("D:/UoM/project/dataset/Babati/1.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
t1 = cv.getTickCount()
# inverse(src)
create_image()
t2 = cv.getTickCount()
time = (t2 - t1)/cv.getTickFrequency()
print("time : %s ms" % (time*1000))
cv.waitKey(0)

cv.destroyAllWindows()
