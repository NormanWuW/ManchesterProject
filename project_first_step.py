import cv2 as cv
import numpy as np


def adjust_contrast(image):
    """g(x,y)=a∗f(x,y)+b"""
    a = 1.1
    b = 80
    rows, cols, channels = image.shape
    cb_dst = image.copy()
    for i in range(rows):
        for j in range(cols):
            for c in range(3):
                color = image[i, j][c] * a + b
                if color > 255:  # 防止像素值越界（0~255）
                    cb_dst[i, j][c] = 255
                elif color < 0:  # 防止像素值越界（0~255）
                    cb_dst[i, j][c] = 0

    cv.imshow('cb_dst', cb_dst)
    return cb_dst


def contrast_img(img1, c, b):  # 亮度就是每个像素所有通道都加上b
    """dst = src1 * alpha + src2 * beta + gamma"""
    rows, cols, channels = img1.shape

    # 新建全零(黑色)图片数组:np.zeros(img1.shape, dtype=uint8)
    blank = np.zeros([rows, cols, channels], img1.dtype)
    dst = cv.addWeighted(img1, c, blank, 1-c, b)
    return dst


def corner_detect(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    float_gray = np.float32(gray)
    dst = cv.cornerHarris(float_gray, 6, 5, 0.04)
    # result is dilated for marking the corners
    dst = cv.dilate(dst, None)
    # Threshold for an optimal value, it may vary depending on the image.
    corner_threshold = 0.01 * dst.max()
    # changing the value of the binary image: set white to pixels which bigger than threshold
    height = dst.shape[0]
    width = dst.shape[1]
    dst_change = dst.copy()
    for h in range(height):
        for w in range(width):
            if dst_change[h, w] > corner_threshold:
                dst_change[h, w] = 1
            else:
                dst_change[h, w] = 0
    cv.imshow("dst_change", dst_change)
    return dst_change


def open_close_demo(binary):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    open_result = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    close_result = cv.morphologyEx(open_result, cv.MORPH_CLOSE, kernel)
    return close_result


def building_detect(image):
    adjust_image = contrast_img(image, 1.2, 0)
    # check out the exceptional color
    # ground: [162, 114, 104] [146, 107, 105]
    hsv = cv.cvtColor(adjust_image, cv.COLOR_BGR2HSV)
    lower_hsv = np.array([100, 43, 46])
    upper_hsv = np.array([124, 255, 255])
    mask = cv.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)

    # dst = cv.bitwise_and(image, image, mask=mask)

    # processing the mask place
    # height = dst.shape[0]
    # width = dst.shape[1]
    # channels = dst.shape[2]
    # for row in range(height):
    #     for col in range(width):
    #         # if not green or desert yellow, then change to white
    #         sum = 0
    #         for c in range(channels):
    #             sum += dst[row, col, c]
    #         if sum != 0:
    #             dst[row][col] = [255, 255, 255]
    # cv.imshow("dst", dst)

    # extracting the bright color for potential roofs
    gray = cv.cvtColor(adjust_image, cv.COLOR_BGR2GRAY)
    cv.imshow("gray image: ", gray)
    ret, binary = cv.threshold(gray, 170, 255, cv.THRESH_BINARY)
    # cv.imshow("binary image: ", binary)

    # merge the two pics of binary
    mimage = cv.bitwise_or(binary, mask)
    corner = corner_detect(image)
    height = corner.shape[0]
    width = corner.shape[1]
    for h in range(height):
        for w in range(width):
            mimage[h, w] += corner[h, w]
    cv.imshow("mimage", mimage)

    # open-closing noise
    open_closing_mimage = open_close_demo(mimage)
    cv.imshow("open_mimage", open_closing_mimage)

    # stroke the lines of the buildings' shape
    contours, heriachy = cv.findContours(open_closing_mimage, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        cv.drawContours(image, contours, i, (0, 0, 255), 1)
    cv.imshow("result", image)


src = cv.imread("D:/UoM/project/dataset/1500m/8.jpg")
# src = cv.imread("D:/UoM/project/dataset/3000m/1.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
print(src.shape)
building_detect(src)
# corner_detect(src)
cv.waitKey(0)
cv.destroyAllWindows()
