'''
client_remote 실행파일 test용
'''
import socket
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
from utils.custom_general import TimeCheck

@torch.no_grad()
def run(weights='runs/train/v5l_results2/weights/best.pt'):
    imgsz = 640
    conf_th = 0.45
    iou_th = 0.45
    max_detect = 1000
    device = ''

    half = False

    # -------------------------

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

    # -------------------------

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'

    w = weights
    stride, names = 64, [f'class{i}' for i in range(10000)]

    # model = attempt_load(w, map_location=device)
    # stride = int(model.stride.max())
    # names = ['K', 'L', 'paper', 'rock', 'scissor', 'W']
    # if half:
        # model.half()
    
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
    tc_fps = TimeCheck(out=True)    # out=True면 time debugging 가능
    delay = 10000
    for path, img, im0s, video_cap in dataset:
        delay = tc_fps.check('0', ret=True)
        print()
        if delay > 1./FPS:
            tc_fps.initial()
            tc = TimeCheck(out=False)
            tc.initial('encode')

            retval, im0_enc = cv2.imencode('.jpg', im0s[0], encode_param)   # frame -> encoded frame
            strData = np.array(im0_enc).tobytes()                           # encoded frame -> bytes
            tc.check('done')

            # 송신
            tc.initial('send')
            sock.send(f'{time_sync():.4f}'.ljust(16).encode())
            sock.send(str(len(strData)).ljust(16).encode())
            sock.send(strData)
            tc.check('done')

        # ======================
        press = cv2.waitKey(1)
        if press == ord('q'):
            break
    
    # 끝났다는 신호 전송 후 소켓 닫기
    sock.send(None)
    sock.close()
    
    cv2.destroyAllWindows()

def main():
    check_requirements(exclude=('tensorboard', 'thop'))
    run()

if __name__ == "__main__":
    main()