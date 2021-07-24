import os
import numpy as np
import cv2
import config as cf

class HandDetection():
    '''
    영상에서 손 이미지를 검출해내는 class
    아래 링크의 yolo-hand-detection 모델을 차용함.
    (링크)
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
        self.net, self.output_layers = self.make_model()

        # 저장할 손 이미지의 크기
        self.height = height
        self.width = width


    def make_model(self):
        net = cv2.dnn.readNet(self.weights, self.cfg)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0]- 1] for i in net.getUnconnectedOutLayers()]
        return net, output_layers


    def video2hand(self, size=320, video_path=0, save_image_dir=False):
        '''
        비디오 -> 원본 이미지 -> 손 이미지 추출 함수

        size: yolo-hand-detection 에서 입력으로 들어가는 이미지 크기
            (320, 416, 608 중 택1)
        video_path: 0이면 webcam video, 특정 파일의 경로이면 해당 영상을 지정한다.
        save_image: 영상에서 얻은 손 이미지를 디렉토리에 저장한다.
        '''
        # set yolo-hand-detection model
        net, output_layers = self.net, self.output_layers

        # ndarray for return
        X_data = np.empty((0, self.height, self.width, 3))

        # get video
        cam = cv2.VideoCapture(video_path)

        currentFrame = 0
        while True:
            success, frame = cam.read()
            
            # frame이 존재하는 경우
            if success:

                # 손 위치 감지하기
                hand_area = self.get_coordinate(size, frame)
                if hand_area == False:
                    continue
                
                # 손 이미지 가져오기
                hand_frame = self.resize_hand(frame, hand_area)
                if hand_frame == False:
                    continue

                # X_data에 손 이미지 하나씩 쌓기
                X_data = np.append(
                    X_data,
                    np.expand_dims(hand_frame, axis=0)  # hand_frame.reshape((1, height, width, 3))
                )
                
                if save_image_dir != False:
                    pass
                    ################################
                    # 대충 이미지 저장하는 코드
                    ################################

                currentFrame += 1

            # frame이 존재하지 않는 경우
            else:
                break
        
        cam.release()
        cv2.destroyAllWindows()

        ###################
        # 미완성
        ###################


    def get_coordinate(self, size, frame):
        '''
        손에 해당하는 좌표를 반환한다.

        size : 320, 416, 608 중 택1
        '''
        # frame의 크기 추출
        height, width, _ = frame.shape

        # yolo-hand-detection 으로 손 위치 감지
        net, output_layers = self.net, self.output_layers
        blob = cv2.dnn.blobFromImage(img, 1/255, (size,size), (0,0,0), True, crop=False)
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
        try:
            target = frame[y_start:y_end, x_start:x_end]
        except:
            # 손 이미지의 좌표가 원본 이미지를 벗어난 경우 예외처리
            return False

        # 의도했던 손 이미지 크기에 맞게 resize해서 반환
        return cv2.resize(target, (self.width, self.height), interpolation=cv2.INTER_CUBIC)



def videos_to_data(height=224, width=224, channel=3, get_npy=True, get_images=False):
    '''
    get_npy=True 기본값
    기본적으로 (X_data, Y_data)를 반환. 각각 ndarray
    그와 동시에 npy 파일을 저장

    height, width, channel은
    X_data의 shape을 결정

    get_images=False
    True로 설정 시 이미지를 preprocess_data/webcam/ 디렉토리에 저장
    height, width 기본값 224
    '''
    #model set
    cfg, weights = dir_item(PATH_YOLO)
    net, output_layers = make_model(weights,cfg)

    #create ndarray as default
    if get_npy:
        X = np.empty((0,height,width,channel))
        y_cnt_lst = []
        y_cnt_temp = 0
    
    #loop for each label
    for label_path in dir_item(PATH_RAW_DATA):
        
        #create folder for images when 'get_images=True'
        if get_images: mkdir_under_path(PATH_PREPROCESS_DATA,label_path[-3:])

        #loop for reading each videos
        for video_path in dir_item(label_path):
            cam = cv2.VideoCapture(video_path)
            currentframe = 0

            while True:
                ret, frame = cam.read()
                if ret:
                    hand_area = get_coordinate(320,frame,net,output_layers)
                    if not hand_area: continue

                    #slice hand area and resize
                    try: resized_frame = resize_hand(frame,hand_area,height,width)
                    except: continue

                    #append data at ndarray as default
                    if get_npy:
                        X = np.append(
                        X, resized_frame.reshape((1,height,width,channel)), axis=0
                        )                    
                    if get_images: #write image when 'get_images=True'
                        write_img(resized_frame,video_path,currentframe)
                    currentframe += 1
                    y_cnt_temp += 1
                else: break
            cam.release()
            cv2.destroyAllWindows()
        
        if get_npy:
            y_cnt_lst.append(y_cnt_temp)
            y_cnt_temp = 0
    if get_npy:
        np.save(f'{os.path.join(PATH_PREPROCESS_DATA,"X_data")}',X)
        Y = onehot_ndarray(y_cnt_lst)
        np.save(f'{os.path.join(PATH_PREPROCESS_DATA,"Y_data")}',Y)
        return X, Y