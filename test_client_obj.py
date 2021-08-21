'''
client_obj 실행파일 test용
'''
import socket
import os
from pathlib import Path

import numpy as np
import cv2

from utils.torch_utils import time_sync
from utils.custom_general import TimeCheck

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
TCP_PORT = 5002
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

    # 데이터 수신
    tc = TimeCheck(out=False)
    tc.initial('receive')
    try:
        t_server = float(recvall(sock_conn, 16))
        t_clnt_remote = float(recvall(sock_conn, 16))
        length = recvall(sock_conn, 16)
        gestures = recvall(sock_conn, int(length))
    except:
        print('break')
        break

    # ping 계산
    t_clnt_obj = time_sync()
    ping_total = t_clnt_obj - t_clnt_remote
    ping_server = t_clnt_obj - t_server

    # 제스쳐 리스트 확인
    gestures = np.frombuffer(gestures, dtype=int)
    if gestures.sum() > 0:
        print(f'[total-{ping_total:7.2f}sec] ' + f'[total-{ping_server:7.2f}sec]', gestures)

    # 수신된 제스쳐에 따라 로직 진행


    # 로직 진행 결과에 따른 원격 조작 실행
    

    cv2.waitKey(1)
sock.close()
