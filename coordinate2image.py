import cv2 as cv
import numpy as np


def coordinate2img(img_path, coordinates, sum):
    extract_img_path = "D:/UoM/project/dataset/counter_extract_img/"
    img = cv.imread(img_path)
    cv.imshow("origin", img)
    count = 0
    for coords in coordinates:
        count += 1
        coords = list(map(int, coords))
        print(coords)
        x, y, width, height = coords
        roi = img[y:y+height, x:x+width]
        # cv.imshow("roi", roi)
        # cv.waitKey(1)
        name = extract_img_path + str(sum) + "_" + str(count) + ".jpg"
        cv.imwrite(name, roi)


f = open('D:/UoM/project/dataset/info.txt')
context = f.readline()
sum = 0
while context:
    sum += 1
    coordinate = context.split()
    img_path = coordinate[0]
    roi = []
    for i in range(2, len(coordinate), 4):
        roi.append(coordinate[i: i + 4])
    coordinate2img(img_path, roi, sum)
    context = f.readline()
cv.destroyAllWindows()


# src = cv.imread("D:/FIRST TIME CHICKEN.png")
# cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# cv.imshow("input image", src)

# print("hi python")
