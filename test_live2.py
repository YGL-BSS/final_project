import numpy as np
import cv2
import time
import os

import modeling

# model load
model = modeling.GestureClassification()
model.load_model()

cap = cv2.VideoCapture(0)


pTime = 0
num_frame = 0
while True:
    success, frame = cap.read()

    if not success: break

    frame = frame[320-150:320+150, 240-200:240+200]

    # # Median Filtering
    # k_size = 33
    # median = cv2.medianBlur(frame, k_size)

    # 예측 결과 출력
    # print(model.predict(np.reshape(frame, (1, 300, 400, 3))))
    print(model.predict(np.expand_dims(frame, axis=0)))

    # 카메라 on
    cv2.imshow('now cam', frame)
    # cv2.imshow('median', median)

    # ESC 누르면 창 닫힘
    key = cv2.waitKey(1) & 0xFF
    if key == 27: break

cap.release()
cv2.destroyAllWindows()

