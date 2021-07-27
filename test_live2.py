import numpy as np
import cv2
import time
import os

import modeling

# model load
model = modeling.GestureClassification()
model.create_model3()
model.load_model()

cap = cv2.VideoCapture(0)

print('가즈아아아아아아')
pTime = 0
num_frame = 0
while True:
    success, frame = cap.read()

    if not success: break

    k = 224//2
    frame = frame[320-k:320+k, 240-k:240+k]

    # # Median Filtering
    # k_size = 33
    # median = cv2.medianBlur(frame, k_size)

    # 예측 결과 출력
    labels = ['빠', '주먹', '총', '오케이', '브이', '롹큰롤']
    if num_frame % 5 == 0:
        result = model.predict(np.expand_dims(frame, axis=0))[0].tolist()
        result_label = labels[result.index(max(result))]
        print(result_label, max(result))


    # 카메라 on
    cv2.imshow('now cam', frame)
    # cv2.imshow('median', median)

    num_frame = (num_frame + 1) % 5

    # ESC 누르면 창 닫힘
    key = cv2.waitKey(1) & 0xFF
    if key == 27: break

cap.release()
cv2.destroyAllWindows()

