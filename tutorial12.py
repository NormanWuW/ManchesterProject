import cv2 as cv
import numpy as np


def template_demo():
    sample = cv.imread("D:/SECOND CHICKEN sample.png")
    target = cv.imread("D:/SECOND CHICKEN.png")
    cv.imshow("sample", sample)
    cv.imshow("target", target)
    methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]
    th, tw = sample.shape[:2]
    for method in methods:
        result = cv.matchTemplate(target, sample, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        print(min_val)
        print(min_loc)
        print(max_val)
        print(max_loc)
        if method == cv.TM_SQDIFF_NORMED:
            tl = min_loc
        else:
            tl = max_loc
        br = (tl[0]+tw, tl[1]+th)
        cv.rectangle(target, tl, br, (0, 0, 255), 2)
        cv.imshow("match-"+np.str(method), target)


# src = cv.imread("D:/FIRST TIME CHICKEN.png")
# cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# cv.imshow("input image", src)
template_demo()
cv.waitKey(0)
cv.destroyAllWindows()