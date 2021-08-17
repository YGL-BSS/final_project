'''
https://walkinpcm.blogspot.com/2016/05/python-python-opencv-tcp-socket-image.html
'''
import socket
import cv2
import numpy as np

# socket 수신 버퍼(인코딩된 이미지)를 읽어서 반환하는 함수
def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None

        buf += newbuf
        count -= len(newbuf)
    
    return buf

# 수신에 사용될 내 ip와 port 번호
TCP_IP = socket.gethostbyname(socket.gethostname())
TCP_PORT = 5001
print('Server IP    :', TCP_IP)
print('Server PORT  :', TCP_PORT)

# TCP 소켓 열고 수신 대기
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((TCP_IP, TCP_PORT))
sock.listen(True)
sock_conn, addr = sock.accept()

print('접속 IP :', addr)

# string 형태의 이미지를 수신받아서 이미지로 변환하고 화면에 출력
while True:
    length = recvall(sock_conn, 16)  # 길이 16의 데이터 먼저 수신 : 이미지 길이를 먼저 수신
    if type(length) == type(None):
        break
    stringData = recvall(sock_conn, int(length))
    data = np.fromstring(stringData, dtype='uint8')
    decimg = cv2.imdecode(data, 1)
    cv2.imshow('Server', decimg)
    cv2.waitKey(1)
sock.close()
cv2.destroyAllWindows()