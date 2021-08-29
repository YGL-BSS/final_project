'''
server 실행파일 test용
'''
from utils.augmentations import letterbox
import numpy as np
import base64

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
from utils.custom_general import TimeCheck, GestureBuffer

from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

weights='runs/train/v5m_results_v03/weights/best.pt'
imgsz = 640
conf_th = 0.55
iou_th = 0.45
max_detect = 1000
device = ''

half = False

print('weights : ', weights)

# Initialize
set_logging()
device = select_device(device)
half &= device.type != 'cpu'

w = weights
stride, names = 64, [f'class{i}' for i in range(10000)]

model = attempt_load(w, map_location=device)
stride = int(model.stride.max())
# names = ['K', 'L', 'paper', 'rock', 'scissor', 'W']
names = ['five', 'four', 'K', 'L', 'one', 'three', 'two', 'zero']
if half:
    model.half()

# Resize image
imgsz = check_img_size(imgsz, s=stride)

# Run Interface
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))) # run once

# custom process
BF = GestureBuffer(names=names)

def detect_gesture(img_bytes, t_send):
    tc = TimeCheck(out=False)
    tc.initial('receive')

    # decode
    im0_temp = cv2.imdecode(np.frombuffer(img_bytes, dtype='uint8'), 1)
    im0s = [im0_temp]
    img = np.stack(im0s, 0)
    img = img[..., ::-1].transpose((0, 3, 1, 2))
    img = np.ascontiguousarray(img)

    #### (구) 코드 진행 ####
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()   # uint8 to fp16/32
    img /= 255.0                                # normalize : 0~255 to 0~1
    
    if len(img.shape) == 3:
        img = img[None] # expand for batch dim

    pred = model(img, augment=False, visualize=False)[0]        # bbox 예측하기
    # NMS
    pred = non_max_suppression(pred, conf_thres=conf_th, iou_thres=iou_th, classes=None, agnostic=False, max_det=max_detect)
    t_process = tc.check('0', ret=True)
    # Detected gesture list
    detected_list = []
    detected_num = [0] * len(names)     # names와 대응하는 list
    # detected_data = {i:[name, 0] for i, name in enumerate(names)}
    for i, detect in enumerate(pred):

        im0 = im0s[i].copy()
        gain = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalize gain whwh

        if len(detect):
            # Rescale boxes from img_size to im0_size
            detect[:, :4] = scale_coords(img.shape[2:], detect[:, :4], im0.shape).round()

            # Print results
            for c in detect[:, -1].unique():
                n = (detect[:, -1] == c).sum()  # detections per class
                detected_num[int(c)] += int(n)

            # Write results
            for *xyxy, conf, cls in reversed(detect):
                c = int(cls)
                label = f'{names[c]} {conf:.2f}'
                plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=3)

    BF.update_buf(t_send, detect_num=detected_num)
    gesture_list = BF.get_action()
    if gesture_list.sum() > 0:
        print(gesture_list)
    
    return gesture_list


def read_buffer(t_send):
    BF.update_buf(t_send)
    gesture_list = BF.get_action()

    return gesture_list


@app.route('/')
def home():
    return "This is homepage"


@app.route('/sendimg', methods=['POST'])
def sendimg():##################이름바꾸기
    if request.method == 'POST':
        img_bytes = request.files['file'].read()
        t_send = float(request.files['t_send'].read())
        try:
            gestures = detect_gesture(img_bytes=img_bytes, t_send=t_send)
            # gestures = gestures.tobytes()       # encode
            result = 'Success'
        except:
            result = 'Fail'
        
        ping = time_sync() - t_send

        return jsonify({'result': result, 'ping': f'{ping:.4f}'})


@app.route('/getcmd', methods=['GET'])
def getcmd():
    if request.method == 'GET':
        gestures = read_buffer(time_sync())
        gestures = gestures.tobytes()

        return jsonify({'gestures': str(gestures)})

if __name__ == '__main__':
    app.run('0.0.0.0', port=38080)