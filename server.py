'''
리모컨 쪽 실행파일
https://walkinpcm.blogspot.com/2016/05/python-python-opencv-tcp-socket-image.html
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

# socket 수신 버퍼(인코딩된 이미지)를 읽어서 반환하는 함수
def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None

        buf += newbuf
        count -= len(newbuf)
    
    return buf

class TimeCheck():
    def __init__(self):
        self.start = '0'
        self.end = '0'
        self.t_start = time.time()
        self.t_end = time.time()

    def initial(self, tag='start'):
        print()
        self.start = tag
        self.t_start = time.time()

    def check(self, tag='end', out=True):
        self.end = tag
        self.t_end = time.time()
        if out:
            time_interval = self.t_end - self.t_start
            print(f'[{self.start} ~ {self.end}] {time_interval:5.2f} sec', end='\t')
            return time_interval

@torch.no_grad()
def run(weights='runs/train/v5l_results2/weights/best.pt'):
    imgsz = 640
    conf_th = 0.45
    iou_th = 0.45
    max_detect = 1000
    device = ''

    half = False

    # 수신에 사용될 내 ip와 port 번호
    TCP_IP = socket.gethostbyname(socket.gethostname())
    TCP_IP = '172.16.6.181'
    TCP_PORT = 5001
    print('Server IP    :', TCP_IP)
    print('Server PORT  :', TCP_PORT)

    # TCP 소켓 열고 수신 대기
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((TCP_IP, TCP_PORT))
    sock.listen(True)
    sock_conn, addr = sock.accept()
    print('접속 IP :', addr)

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'

    w = weights
    stride, names = 64, [f'class{i}' for i in range(10000)]

    model = attempt_load(w, map_location=device)
    stride = int(model.stride.max())
    names = ['K', 'L', 'paper', 'rock', 'scissor', 'W']
    if half:
        model.half()
    
    # Resize image
    imgsz = check_img_size(imgsz, s=stride)

    # # Dataloader
    # view_img = check_imshow()
    # cudnn.benchmark = True  # set True to speed up constant image size inference
    # dataset = LoadStreams('0', img_size=imgsz, stride=stride)
    # bs = len(dataset)

    # video_path, video_writer = [None] * bs, [None] * bs

    # Run Interface
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))) # run once
    
    # custom process
    EI = EncodeInput(50)
    G2C = Gesture2Command()

    # t0 = time.time()

    # ==================================

    tc = TimeCheck()
    # string 형태의 이미지를 수신받아서 이미지로 변환하고 화면에 출력
    while True:
        # # img
        # length = recvall(sock_conn, 16)     # 길이 16의 데이터 먼저 수신 : 이미지 길이를 먼저 수신
        # if type(length) == type(None):
        #     break
        # stringImg_shape = recvall(sock_conn, 16)
        # stringImg = recvall(sock_conn, int(length))
        # #img = torch.tensor(np.frombuffer(stringImg, dtype=np.float32))    # string -> tensor
        # img = np.frombuffer(stringImg, dtype=np.float32)    # string -> tensor
        # #print(img.shape, stringImg_shape.decode())
        # img = img.reshape(tuple([int(i) for i in stringImg_shape.decode().strip().split()]))

        # # im0s
        # stringim0s_length = recvall(sock_conn, 10)     # 길이 16의 데이터 먼저 수신 : 이미지 길이를 먼저 수신
        # stringim0s_length = int(stringim0s_length.decode().strip())
        # im0s = []
        # for _ in range(stringim0s_length):
        #     length = recvall(sock_conn, 16)
        #     stringIm0 = recvall(sock_conn, int(length))
        #     im0s.append(
        #         cv2.imdecode(np.fromstring(stringIm0, dtype='uint8'), 1)
        #     )
        
        # 소켓에서 데이터 받기
        tc.initial('0')
        length = recvall(sock_conn, 16)     # 길이 16의 데이터 먼저 수신 : 이미지 길이를 먼저 수신
        if type(length) == type(None):
            break
        tc.check('1')
        stringImg_shape = recvall(sock_conn, 16)
        tc.check('2')
        stringImg = recvall(sock_conn, int(length))
        tc.check('3')

        # 받은 데이터를 통해 img, im0s 복구하기
        img_shape = tuple([int(n) for n in stringImg_shape.decode().strip().split()])
        # img = np.frombuffer(stringImg, dtype=np.float32).copy()    # string -> tensor
        img = np.frombuffer(base64.b64decode(stringImg), dtype=np.float32).copy()
        img *= 255.0
        img = img.reshape(img_shape)
        img = img.astype('uint8')
        
        im0_temp = img.copy()
        im0_temp = im0_temp.transpose((0, 2, 3, 1))     # (1, 3, 480, 640) -> (1, 480, 640, 3)
        im0_temp = im0_temp[..., ::-1]                  # RGB -> BGR
        im0_temp = im0_temp.reshape(im0_temp.shape[-3:])
        im0s = [im0_temp]
        # print(img[0,:,0,0], im0s[0][0,0,:])
        tc.check('4')

        # (구) 코드 진행
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()   # uint8 to fp16/32
        img /= 255.0                                # normalize : 0~255 to 0~1
        
        if len(img.shape) == 3:
            img = img[None] # expand for batch dim

        t1 = time_sync() 
        pred = model(img, augment=False, visualize=False)[0]
        tc.check('5')

        # NMS
        pred = non_max_suppression(pred, conf_thres=conf_th, iou_thres=iou_th, classes=None, agnostic=False, max_det=max_detect)
        t2 = time_sync()
        tc.check('6')

        # Detected gesture list
        detected_list = []
        detected_num = []

        for i, detect in enumerate(pred):
            # p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            # p = Path(p) # to Path
            # save_path
            # txt_path
            s, im0 = f'{i}: ', im0s[i].copy()

            s += '%gx%g ' % img.shape[2:]
            gain = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalize gain whwh
            # im_cp = im0.copy()    # for save_crop

            if len(detect):
                # Rescale boxes from img_size to im0_size
                detect[:, :4] = scale_coords(img.shape[2:], detect[:, :4], im0.shape).round()

                # Print results
                for c in detect[:, -1].unique():
                    n = (detect[:, -1] == c).sum()  # detections per class
                    s += f'{n} {names[int(c)]} '    # add to string
                    detected_num.append(f'{n}')
                    detected_list.append(names[int(c)])

                # Write results
                for *xyxy, conf, cls in reversed(detect):
                    c = int(cls)
                    label = f'{names[c]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=3)

        # Print time (inference + NMS)
        # print(f'{s} Done. ({t2 - t1:.3f}s)', end='\t')

        # Get bbox list                     # 전유상 팀원 코드 추가
        hand_detected = []
        for n, label in zip(detected_num, detected_list):
            hand_detected += [label] * int(n)

        # preprocess detected hand signal   # 정민형 팀원 코드 추가
        cmd = EI.encode(hand_detected)

        # Action according to the command   # 박도현 팀원 코드 추가
        if cmd:
            print(cmd)
            G2C.activate_command(cmd)

        # show fps
        cv2.putText(
            im0, f'FPS: {1/(t2 - t1):.2f}', (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
        )
        tc.check('7')
        # Stream
        cv2.imshow('webcam', im0)
        cv2.waitKey(1)


    sock.close()
    cv2.destroyAllWindows()

    # ==================================


def main():
    check_requirements(exclude=('tensorboard', 'thop'))
    run()

if __name__ == "__main__":
    main()

