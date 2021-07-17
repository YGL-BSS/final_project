import cv2
import numpy as np

def cont():
    try:
        cap=cv2.VideoCapture(0)
    except:
        print('camera_error')
        return

    t = 0
    while t < 100 * 60:
        ret, frame = cap.read()

        if not ret:
            print('camera2_error')
            break
        
        dst = frame.copy()
        test = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask_hand = cv2.inRange(test, np.array([0,133,77]), np.array([255,173,127]))
        test = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thr = cv2.threshold(mask_hand, 127, 255, cv2.THRESH_BINARY_INV)

        canny = cv2.Canny(dst, 50, 150)

        contours, hierachy = cv2.findContours(thr, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        for i in contours:
            hull = cv2.convexHull(i,clockwise=True)
            cv2.drawContours(dst, [hull], 0, (0,0,255), 2)

        cv2.imshow('dst', dst)
        cv2.imshow('mask_hand', mask_hand)
        cv2.imshow('canny', canny)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        
        if t % 100 == 0:
            print(t)
        t += 1

    cap.release()
    cv2.destroyAllWindows()



# src = cv2.imread('building.jpg', cv2.IMREAD_GRAYSCALE)

# if src is None:
#     print('Image load failed!')
#     sys.exit()



# cv2.imshow('src', src)
# cv2.imshow('dst', dst)
# cv2.waitKey()

# cv2.destroyAllWindows()