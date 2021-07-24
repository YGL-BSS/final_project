import os
import numpy as np
import cv2
import time

import config as cf

class HandDetection():
    '''
    영상에서 손 이미지를 검출해내는 class
    아래 링크의 yolo-hand-detection 모델을 차용함.
        https://github.com/cansik/yolo-hand-detection
    
    # --------- class 사용 방법 예시 ---------------------
    case 1. Webcam에서 손 이미지 데이터를 얻는 경우
        hand = HandDetection(selected_model='v3')
        hand.video2hand(save_image_dir='label_001')
    
    case 2. 저장된 video에서 손 이미지 데이터를 얻는 경우
        hand = HandDetection(selected_model='v3')
        hand.video2hand(video_path='./video.mp4', save_image_dir='label_001')
    # ---------------------------------------------------
    '''
    model_dict = {
        'v3': 'cross-hands',
        'v3-prn': 'cross-hands-tiny-prn',
        'v4': 'cross-hands-yolov4-tiny'
    }

    def __init__(self, selected_model='v3', height=224, width=224):

        # 불러올 yolo-hand-detection 모델 정보 파일의 위치 지정
        self.weights = os.path.join(cf.PATH_YOLO, f'{self.model_dict[selected_model]}.weights')
        self.cfg = os.path.join(cf.PATH_YOLO, f'{self.model_dict[selected_model]}.cfg')

        # 불러온 yolo-hand-detection 모델
        print('Load yolo model...', end='')
        self.net, self.output_layers = self.get_yolo_model()
        print('Done!\n')

        # 저장할 손 이미지의 크기
        self.height = height
        self.width = width


    def get_yolo_model(self):
        net = cv2.dnn.readNet(self.weights, self.cfg)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0]- 1] for i in net.getUnconnectedOutLayers()]
        return net, output_layers


    def video2hand(self, yolo_size=320, video_path=0, save_image_dir=False):
        '''
        비디오 -> 원본 이미지 -> 손 이미지 추출 함수

        size: yolo-hand-detection 에서 입력으로 들어가는 이미지 크기
            (320, 416, 608 중 택1)
        video_path: 0이면 webcam video, 특정 파일의 경로이면 해당 영상을 지정한다.
        save_image: 영상에서 얻은 손 이미지를 디렉토리에 저장한다.
        '''
        # # set yolo-hand-detection model
        # net, output_layers = self.net, self.output_layers

        # ndarray for return
        X_data = np.empty((0, self.height, self.width, 3))

        # get video
        cam = cv2.VideoCapture(video_path)

        # current Frame number
        currentFrame = 0
        folder = cf.mkdir_under_path(cf.PATH_PREPROCESS_DATA, save_image_dir)
        for frame_num in [f[:-4] for f in os.listdir(folder) if f.endswith(r".jpg")]:
            currentFrame = max(currentFrame, int(frame_num))
        currentFrame += 1

        time_start = time.time()
        while True:
            # 키 입력시 종료
            if (video_path == 0) and (cv2.waitKey(20) == ord('q')):
                break

            success, frame = cam.read()

            clear()
            print(f'사진 저장 디렉토리 : {folder}')
            print(f'frame no. {currentFrame:0>5d}')
            print(f'실행시간 : {time.time() - time_start:.2f}초')
            
            # frame이 존재하는 경우
            if success:

                # 손 위치 감지하기
                hand_area = self.get_coordinate(yolo_size, frame)
                if hand_area == False:
                    continue
                
                # 손 이미지 가져오기
                try:
                    hand_frame = self.resize_hand(frame, hand_area)
                except:
                    continue

                # X_data에 손 이미지 하나씩 쌓기
                X_data = np.append(
                    X_data,
                    np.expand_dims(hand_frame, axis=0)  # hand_frame.reshape((1, height, width, 3))
                )
                
                # 사진 저장하기
                if save_image_dir != False:
                    cv2.imwrite(
                        os.path.join(folder, f'{currentFrame:0>5d}.jpg'),
                        # os.path.join(folder, f'{save_image_dir}_{currentFrame}.jpg'),
                        hand_frame
                    )

                # Frame 번호 +1
                currentFrame += 1
                
                # 시각화
                # cv2.imshow('Frame', hand_frame)

            # frame이 존재하지 않는 경우
            else:
                print('완료!')
                time.sleep(3)
                break

        cam.release()
        # cv2.destroyAllWindows()

        return X_data

    #########################################################################

    def get_coordinate(self, size, frame):
        '''
        손에 해당하는 좌표를 반환한다.

        size : 320, 416, 608 중 택1
        '''
        # frame의 크기 추출
        height, width, _ = frame.shape

        # yolo-hand-detection 으로 손 위치 감지
        net, output_layers = self.net, self.output_layers
        blob = cv2.dnn.blobFromImage(frame, 1/255, (size,size), (0,0,0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # 감지된 좌표 저장
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Hand detect
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle x_start and y_start
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h]) 
        
        # 감지된 좌표 하나만 출력
        if len(boxes):
            return boxes[0]
        else:
            return False


    def resize_hand(self, frame, hand_area, alpha=0.1):
        '''
        원본 이미지와 손에 해당하는 좌표를 받은 뒤, 손 이미지로 편집하여 반환하는 함수
        frame에서 hand_area 의 x, y, w, h를 참조한다.
        '''
        x, y, w, h = hand_area

        # 원본 이미지에서 손에 해당하는 좌표 계산
        y_start = round(y * (1-alpha))
        x_start = round(x * (1-alpha))
        y_end = round((y+h) * (1+alpha))
        x_end = round((x+w) * (1+alpha))

        # 원본 이미지에서 손 이미지 슬라이싱
        target = frame[y_start:y_end, x_start:x_end]

        # 의도했던 손 이미지 크기에 맞게 resize해서 반환
        return cv2.resize(target, (self.width, self.height), interpolation=cv2.INTER_CUBIC)



def clear():
    os.system('cls' if os.name == 'nt' else 'clear')