'''
https://walkinpcm.blogspot.com/2016/05/python-python-opencv-tcp-socket-image.html
'''
import socket
import cv2
import numpy as np

# 연결할 서버의 ip주소와 port번호
# https://whitewing4139.tistory.com/103
TCP_IP = input('Server IP >> ')
TCP_PORT = 5001
print('Server IP    :', TCP_IP)
print('Server PORT  :', TCP_PORT)

# 송신을 위한 socket 생성
sock = socket.socket()
sock.connect((TCP_IP, TCP_PORT))
print(f'Connected to {TCP_IP}:{TCP_PORT}')

# OpenCV를 이용해서 webcam으로부터 이미지 추출
cam = cv2.VideoCapture(0)
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
while True:
    success, frame = cam.read()
    if not success:
        sock.send(None)
        break

    # 추출한 이미지를 string 형태로 인코딩시키는 과정
    retval, imgencode = cv2.imencode('.jpg', frame, encode_param)   # result: boolean, imgencode.shape=(??, 1)
    data = np.array(imgencode)
    stringData = data.tostring()

    # string 형태로 인코딩한 이미지를 socket을 통해서 전송
    sock.send(str(len(stringData)).ljust(16).encode())   # 좌로 밀기
    sock.send(stringData)
    # decimg = cv2.imdecode(data, 1)
    # cv2.imshow('Client', decimg)
    # cv2.waitKey(2000)

sock.close()
cv2.destroyAllWindows()
