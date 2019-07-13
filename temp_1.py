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
    coordinate = []
    for h in range(height):
        for w in range(width):
            if dst_change[h, w] > corner_threshold:
                dst_change[h, w] = 1
                coordinate.append((h, w))
            else:
                dst_change[h, w] = 0
    cv.imshow("dst_change", dst)
    print(coordinate)
    return dst_change, coordinate


def open_close_demo(binary):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    open_result = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    close_result = cv.morphologyEx(open_result, cv.MORPH_CLOSE, kernel)
    return close_result


def threshold_demo(gray):
    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    print("threshold value %s" % ret)
    cv.imshow("binary", binary)
    return binary


def local_threshold_demo(gray):
    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 25, 10)
    cv.imshow("binary", binary)
    return binary


def custom_threshold(gray):
    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    m = np.reshape(gray, [1, w * h])
    mean = m.sum() / (w*h)
    print("mean:", mean)
    ret, binary = cv.threshold(gray, mean, 255, cv.THRESH_BINARY)
    print("threshold value:", ret)
    cv.imshow("binary", binary)
    return binary


def line_detect_possible_demo(gray):
    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    # cv.imshow("edge", edges)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=20, maxLineGap=5)
    return lines


def building_detect(image):
    adjust_image = contrast_img(image, 1.2, 0)
    # extracting the bright color for potential roofs
    gray = cv.cvtColor(adjust_image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    binary = custom_threshold(gray)
    cv.imshow("binary image: ", binary)
    # lines = line_detect_possible_demo(gray)
    # for line in lines:
    #     x1, y1, x2, y2, = line[0]
    #     cv.line(gray, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # cv.imshow("image_edge", gray)
    corner, coordinates = corner_detect(image)
    for coordinate in coordinates:
        cv.circle(image, coordinate, 10, (0, 0, 255), -1)
    cv.imshow("corner", corner)
    # height = corner.shape[0]
    # width = corner.shape[1]
    # for h in range(height):
    #     for w in range(width):
    #         mimage[h, w] += corner[h, w]
    # cv.imshow("mimage", mimage)
    #
    # open-closing noise
    open_closing_image = open_close_demo(binary)
    # cv.imshow("open_mimage", open_closing_image)

    # stroke the lines of the buildings' shape
    contours, heriachy = cv.findContours(open_closing_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
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
