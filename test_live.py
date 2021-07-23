import cv2
import time
import os

cap = cv2.VideoCapture(0)
# wCam, hCam = 640, 480
# cap.set(3, wCam)
# cap.set(4, hCam)

pTime = 0
num_frame = 0
while True:
    success, frame = cap.read()

    if not success: break

    frame = cv2.resize(frame, dsize=(224, 224), interpolation=cv2.INTER_AREA)

    # Median Filtering
    k_size = 33
    median = cv2.medianBlur(frame, k_size)

    print(frame.shape, median.shape)

    # # Average Filtering
    # k_size = 33
    # avg = cv2.blur(frame, (k_size, k_size))

    # Convolution Filtering
    # conv = cv2.filter2D(frame, )

    # fps 표시
    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
    3, (255, 0, 0), 3)

    # # 저장
    # cv2.imwrite(f'save_picture/{num_frame}.jpg', frame)
    # num_frame += 1

    # 카메라 on
    cv2.imshow('now cam', frame)
    cv2.imshow('median', median)

    # ESC 누르면 창 닫힘
    key = cv2.waitKey(1) & 0xFF
    if key == 27: break

cap.release()
cv2.destroyAllWindows()

