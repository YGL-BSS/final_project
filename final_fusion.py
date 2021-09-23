'''
client_obj 실행파일 test용
'''
import os
from pathlib import Path

import numpy as np
import cv2

import custom_ppt

import winreg

from utils.torch_utils import time_sync
from utils.custom_general import TimeCheck

import requests

# 사용가능 명령어 및 제스쳐 리스트
command_lists = ['StartEnd', 'NextSlide', 'PreviousSlide', 'AddMedia', 'AddLink']
motion_lists = ['five', 'four' ,'K', 'L', 'one', 'three', 'two', 'zero']

# 저장된 파일의 확장자 명에 따라 윈도우즈에서 설정되어있는 
# 기본 실행파일 경로 탐색 및 저장
def find_exe(file):
    extension = '.' + file.split('.')[-1]
    exe = winreg.QueryValue(winreg.HKEY_CLASSES_ROOT, extension)
    created_key = winreg.CreateKey(winreg.HKEY_CLASSES_ROOT, exe)
    exepath = winreg.QueryValue(created_key, 'DefaultIcon')
    exepath = exepath.split(',')[0]
    if exepath[0] in ["'",'"']:
        exepath = eval(exepath)
    return exepath

# 들어온 제스쳐에 따라 명령 실행
def Gesture2Command(class_lists, given_motion):
    for class_list in class_lists:
        class_list.g2c(given_motion)

# custom으로 만들어진 list를 알맞은 형태로 변형
def list2para(custom):
    if custom[0] in command_lists[1:3]:
        temp = [custom[0], custom[1:]]
        
    elif custom[0] == command_lists[3] or custom[0] == command_lists[0]:
        exepath = find_exe(custom[-1])
        custom[-1] = custom[-1].replace('/', '\\')
        temp = [custom[0], custom[1:-2], exepath, custom[-1]]

    elif custom[0] == command_lists[4]:
        temp = [custom[0], custom[1:-1], custom[-1]]

    return temp

# ppt_elements 폴더 경로 설정
PATH_BASE = os.path.dirname(__file__)
ppt_elements_path = os.path.join(PATH_BASE, 'ppt_elements')
ppt_elements = os.listdir(ppt_elements_path)

# 저장된 제스쳐 및 명령을 알맞은 형태로 변환
customPPT = 'customPPT.txt'

with open(os.path.join(ppt_elements_path, customPPT)) as f:
    customs = [list2para(l[:-1].split('|')) for l in f.readlines()]

# 변환된 명령어를 통해 사용되는 클래스들을 리스트에 저장
class_lists = []

for custom in customs:
    class_lists.append(getattr(custom_ppt, custom[0])(*custom[1:]))

# server 주소
SERVER_IP = input('server의 ip를 입력해주세요 >> ')
# SERVER_IP = '222.111.51.152'
URI = f'http://{SERVER_IP}:38080/getcmd'

while True:

    res = requests.get(URI)
    res = res.json()
    gestures = np.frombuffer(eval(res['gestures']), dtype=np.int32)

    ############################################

    # 입력된 gesture에 따라 command 수행하면 됨.
    if gestures.sum() > 0:
        Gesture2Command(class_lists, gestures)
        # print(gestures.tolist(), motion_lists[gestures.tolist().index(1)])
        

    ############################################

    if cv2.waitKey(10) == ord('q'):
        break

print('Fin.')