import imutils
import numpy as np
import cv2

lower = np.array([0,125,77], dtype = "uint8")
upper = np.array([255, 200, 150], dtype = "uint8")

# 전체 경계 중 넓이가 가장 큰 경계 사각형을 반환
def biggestContour(contours):
    x = 0
    y = 0
    w = 0
    h = 0
    contour = []
    for temp_contour in contours:
        temp_x,temp_y,temp_w,temp_h = cv2.boundingRect(temp_contour)
        if w*h < temp_w * temp_h:
            x = temp_x
            y = temp_y
            w = temp_w
            h = temp_h

    return (x, y, w, h)

# binary image 추출 = mask
def makeMask(frame):
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skinMask = cv2.inRange(converted, lower, upper)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,3))
    skinMask = cv2.erode(skinMask, kernel, iterations = 6)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,2))
    skinMask = cv2.dilate(skinMask, kernel, iterations = 1)
    
    skinMask = cv2.GaussianBlur(skinMask, (3,3), 0)

    return skinMask

# 해당 이미지의 손 좌표를 반환
def makeRec(frame):
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skinMask = cv2.inRange(converted, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
    
    skinMask = cv2.GaussianBlur(skinMask, (3,3), 0)

    contours,_ = cv2.findContours(skinMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return biggestContour(contours)