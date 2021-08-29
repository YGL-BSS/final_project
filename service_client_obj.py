'''
client_obj 실행파일 test용
'''
import os
from pathlib import Path

import numpy as np
import cv2

from utils.torch_utils import time_sync
from utils.custom_general import TimeCheck

import requests

# server 주소
SERVER_IP = input('server의 ip를 입력해주세요 >> ')
# SERVER_IP = '222.111.51.152'
URI = f'http://{SERVER_IP}:38080/getcmd'

names = ['five', 'four' ,'K', 'L', 'one', 'three', 'two', 'zero']
while True:

    res = requests.get(URI)
    res = res.json()
    gestures = np.frombuffer(eval(res['gestures']), dtype=np.int32)

    ############################################

    # 입력된 gesture에 따라 command 수행하면 됨.
    if gestures.sum() > 0:
        print(gestures)
        # print(gestures.tolist(), names[gestures.tolist().index(1)])
        

    ############################################

    if cv2.waitKey(10) == ord('q'):
        break

print('Fin.')
