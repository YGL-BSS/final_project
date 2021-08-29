'''
client_remote 실행파일 test용
'''
import numpy as np

from pathlib import Path
import sys
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())

import cv2
import time
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams
from utils.general import check_requirements, check_img_size, check_imshow, set_logging, increment_path, non_max_suppression, \
    scale_coords
from utils.plots import plot_one_box, colors
from utils.torch_utils import select_device, time_sync

# custom module
from utils.ppt import output_to_detect, EncodeInput, Gesture2Command
from utils.custom_general import TimeCheck

import requests
import os

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

# server 주소
SERVER_IP = input('server의 ip를 입력해주세요 >> ')
# SERVER_IP = '222.111.51.152'
URI = f'http://{SERVER_IP}:38080/sendimg'

imgsz = 640
device = ''
half = False

# Initialize
set_logging()
device = select_device(device)
half &= device.type != 'cpu'

# w = weights
stride, names = 64, [f'class{i}' for i in range(10000)]

# Resize image
imgsz = check_img_size(imgsz, s=stride)

# Dataloader
view_img = check_imshow()
cudnn.benchmark = True  # set True to speed up constant image size inference
dataset = LoadStreams('0', img_size=imgsz, stride=stride)
bs = len(dataset)

video_path, video_writer = [None] * bs, [None] * bs

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
FPS = 30
tc_fps = TimeCheck(out=False)    # out=True면 time debugging 가능
delay = 10000
cnt = 0
for path, img, im0s, video_cap in dataset:
    delay = tc_fps.check('0', ret=True)
    if delay > 1./FPS:
        cnt += 1
        tc_fps.initial()
        tc = TimeCheck(out=False)
        tc.initial('encode')

        retval, im0_enc = cv2.imencode('.jpg', im0s[0], encode_param)   # frame -> encoded frame
        strData = np.array(im0_enc).tobytes()                           # encoded frame -> bytes
        tc.check('done')

        # 송신
        res = requests.post(
            URI,
            files={"file": strData, "t_send": f'{time_sync():.4f}'}
        )

        # res = res.content.decode()
        # if res != 'Success':            # error
        #     print(res, f'(time : {time.localtime()})')
        
        res = res.json()
        if cnt > 10:
            clear()
            now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            # print(f"ping : {float(res['ping']):7.4f} [ms], now : {now}")
            print(f"[{now}] result: {res['result']}\tping: {float(res['ping']):7.4f} sec")
            cnt = 0
        
        cv2.imshow('webcam', im0s[0])
        cv2.waitKey(1)
        
    # ======================
    press = cv2.waitKey(1)
    if press == ord('q'):
        break

cv2.destroyAllWindows()
