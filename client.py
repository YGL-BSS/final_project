'''
https://walkinpcm.blogspot.com/2016/05/python-python-opencv-tcp-socket-image.html
프레젠테이션 실행 파일
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

    # # OpenCV를 이용해서 webcam으로부터 이미지 추출
    # cam = cv2.VideoCapture(0)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    # while True:
    #     success, frame = cam.read()
    #     if not success:
    #         sock.send(None)
    #         break

    #     # 추출한 이미지를 string 형태로 인코딩시키는 과정
    #     retval, imgencode = cv2.imencode('.jpg', frame, encode_param)   # result: boolean, imgencode.shape=(??, 1)
    #     data = np.array(imgencode)
    #     stringData = data.tostring()

    #     # string 형태로 인코딩한 이미지를 socket을 통해서 전송
    #     sock.send(str(len(stringData)).ljust(16).encode())   # 좌로 밀기
    #     sock.send(stringData)
    #     # decimg = cv2.imdecode(data, 1)
    #     # cv2.imshow('Client', decimg)
    #     # cv2.waitKey(2000)

    # sock.close()
    # cv2.destroyAllWindows()

    # -------------------------

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'

    w = weights
    stride, names = 64, [f'class{i}' for i in range(10000)]

    # model = attempt_load(w, map_location=device)
    # stride = int(model.stride.max())
    names = ['K', 'L', 'paper', 'rock', 'scissor', 'W']
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

    # # Run Interface
    # if device.type != 'cpu':
    #     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))) # run once
    
    # custom process
    # EI = EncodeInput(50)
    # G2C = Gesture2Command()

    # t0 = time.time()
    tc = TimeCheck()
    for path, img, im0s, video_cap in dataset:
        tc.initial('0')
        # print(img[0,:,0,0], im0s[0][0, 0, :])
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()   # uint8 to fp16/32
        img /= 255.0                                # normalize : 0~255 to 0~1
        
        if len(img.shape) == 3:
            img = img[None] # expand for batch dim

        # ======================
        # 보낼 string 데이터 만들기
        tc.check('1')
        stringImg_shape = ' '.join([str(i) for i in img.shape])
        # stringImg = img.cpu().numpy().tobytes()      # tensor -> bytes
        stringImg = base64.b64encode(img.cpu().numpy().tobytes())

        # 송신
        tc.check('2')
        sock.send(str(len(stringImg)).ljust(16).encode())
        tc.check('3')
        sock.send(stringImg_shape.ljust(16).encode())
        tc.check('4')
        sock.send(stringImg)
        tc.check('5')
        

    #     # string 형태로 인코딩한 이미지를 socket을 통해서 전송
    #     sock.send(str(len(stringData)).ljust(16).encode())   # 좌로 밀기
    #     sock.send(stringData)
    #     # decimg = cv2.imdecode(data, 1)
    #     # cv2.imshow('Client', decimg)
    #     # cv2.waitKey(2000)


        # ======================
        
        # t1 = time_sync() 
        # pred = model(img, augment=False, visualize=False)[0]

        # # NMS
        # pred = non_max_suppression(pred, conf_thres=conf_th, iou_thres=iou_th, classes=None, agnostic=False, max_det=max_detect)
        # t2 = time_sync()

        # # Detected gesture list
        # detected_list = []
        # detected_num = []

        # for i, detect in enumerate(pred):
        #     p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
        #     p = Path(p) # to Path
        #     # save_path
        #     # txt_path

        #     s += '%gx%g ' % img.shape[2:]
        #     gain = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalize gain whwh
        #     # im_cp = im0.copy()    # for save_crop

        #     if len(detect):
        #         # Rescale boxes from img_size to im0_size
        #         detect[:, :4] = scale_coords(img.shape[2:], detect[:, :4], im0.shape).round()

        #         # Print results
        #         for c in detect[:, -1].unique():
        #             n = (detect[:, -1] == c).sum()  # detections per class
        #             s += f'{n} {names[int(c)]} '    # add to string
        #             detected_num.append(f'{n}')
        #             detected_list.append(names[int(c)])

        #         # Write results
        #         for *xyxy, conf, cls in reversed(detect):
        #             c = int(cls)
        #             label = f'{names[c]} {conf:.2f}'
        #             plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=3)

        # Print time (inference + NMS)
        # print(f'{s} Done. ({t2 - t1:.3f}s)', end='\t')

        # # Get bbox list                     # 전유상 팀원 코드 추가
        # hand_detected = []
        # for n, label in zip(detected_num, detected_list):
        #     hand_detected += [label] * int(n)

        # # preprocess detected hand signal   # 정민형 팀원 코드 추가
        # cmd = EI.encode(hand_detected)

        # # Action according to the command   # 박도현 팀원 코드 추가
        # if cmd:
        #     print(cmd)
        #     G2C.activate_command(cmd)

        # # show fps
        # cv2.putText(
        #     im0, f'FPS: {1/(t2 - t1):.2f}', (10, 50),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
        # )

        # Stream
        # cv2.imshow(str(p), im0)
        # cv2.imshow('webcam', im0)
        cv2.waitKey(1)
    
    # 끝났다는 신호 전송 후 소켓 닫기
    sock.send(None)
    sock.close()
    
    cv2.destroyAllWindows()

def main():
    check_requirements(exclude=('tensorboard', 'thop'))
    run()

if __name__ == "__main__":
    main()