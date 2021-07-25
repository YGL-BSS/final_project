# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from config import *



def make_model(weights, cfg):
    net = cv2.dnn.readNet(weights, cfg)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0]- 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

# cfg, weights = dir_item(PATH_YOLO)
# print(weights,cfg)
# model = make_model(weights,cfg) #test 통과

def get_coordinate(size, img, net, output_layers):
    '''
    size in [320, 416, 608]
    choose 320 for standard
    '''
    # 두번째 매개변수
    #path를 받는 경우 parameter: img_path
    #img = cv2.imread(img_path)
    
    #바로 이미지를 받는 경우
    # parameter : img
    
    height, width, _ = img.shape 
    blob = cv2.dnn.blobFromImage(img, 0.00392,(size,size), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # 감지된 좌표 저장
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

                coordinate = [x, y, w, h]
                return coordinate
    return False

def resize_hand(frame, hand_area, height, width, alpha=0.1):
    '''
    frame 에서 [x,y,w,h]값을 참조,
    alpha 값만큼 늘리거나 줄인 영역을 슬라이싱.
    height,widht 사이즈로 리사이즈하여 반환한다.
    '''
    x, y, w, h = hand_area
    y_start = round(y * (1-alpha))
    x_start = round(x * (1-alpha))
    y_end = round((y+h) * (1+alpha))
    x_end = round((x+w) * (1+alpha))
    target = frame[y_start:y_end, x_start:x_end]
    return cv2.resize(target, (width, height), interpolation=cv2.INTER_CUBIC)

def write_img(img, src_video_name, frame_num):
    '''
    raw_data 의 src_video_name을 참조하여
    img(ndarray)를 preprocess_data의
    해당 레이블 디렉토리에 저장한다.
    몇번째 frame인지 frame_num에 입력
    '''
    folder = os.path.join(PATH_PREPROCESS_DATA, src_video_name[-11:-8])
    name = f'{src_video_name[-7:-4]}_{frame_num:0>5}.jpg'
    cv2.imwrite(os.path.join(folder,name), img)

def onehot_ndarray(y_cnt_lst):
    '''
    예시) [333,222,550]이라는 리스트를 매개변수로 전달할 경우
    위에서부터 차례로
    [1 0 0]  333개
    [0 1 0] 222개
    [0 0 1] 550개
    append된 2차원 ndarray를 반환. 이경우 shape는 (1105,3)
    [10,20,30,40]의 경우 위에서부터
    [1 0 0 0] 10개
    [0 1 0 0] 20개
    [0 0 1 0] 30개
    [0 0 0 1] 40개
    shape: (100,4)
    '''
    Y = np.empty((0,len(y_cnt_lst)),dtype=int)
    for i in range(len(y_cnt_lst)):
            temp = [0] * len(y_cnt_lst)
            temp[i] = 1
            for j in range(y_cnt_lst[i]):
                y_temp = np.reshape(temp,(1, len(y_cnt_lst)))
                Y = np.append(Y,y_temp,axis=0)
    return Y


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

def webcam_to_data(height=224, width=224, channel=3, get_npy=True, get_images=False):
    '''
    get_npy=True 기본값
    기본적으로 X_data를 반환. (ndarray)
    그와 동시에 npy 파일을 저장

    height, width, channel은
    X_data의 shape을 결정

    get_images=False
    True로 설정 시 이미지를 각label 디렉토리에 저장
    height, width 기본값 224
    '''
    #model set
    cfg, weights = dir_item(PATH_YOLO)
    net, output_layers = make_model(weights,cfg)

    #create ndarray as default
    if get_npy: X = np.empty((0,height,width,channel))
    
    #create folder for images when 'get_images=True'
    if get_images: folder_path = mkdir_under_path(PATH_PREPROCESS_DATA,'webcam')

    cam = cv2.VideoCapture(0)
    currentframe = 0

    while cv2.waitKey(33) != ord('q'):
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
                filename = os.path.join(folder_path, f'{currentframe:0>5}.jpg')
                cv2.imwrite(filename, resized_frame)
            currentframe += 1
            cv2.imshow('Frame', resized_frame)
        else: break
    cam.release()
    cv2.destroyAllWindows()

    if get_npy:
        np.save(f'{os.path.join(PATH_PREPROCESS_DATA,"X_data")}',X)
        return X

def model(frame):
    return True

def start_motion(frame, hand_area):
    return model(resize_hand(frame, hand_area, 300,400, 0))
        
def fix_rectangle():

    cfg, weights = dir_item(PATH_YOLO)
    net, output_layers = make_model(weights, cfg)
    cam = cv2.VideoCapture(0)
    find_coordinate = 0
    while cv2.waitKey(33) != ord('q'):
        success, frame = cam.read()
        if success:
            if find_coordinate < 15:
                coordinate = get_coordinate(320, frame, net,output_layers)
                if coordinate and start_motion(frame, coordinate):                    
                    find_coordinate += 1
            
            elif find_coordinate == 15:
                x, y, w, h = coordinate
                find_coordinate += 1
                print('Hand Detected!!!!!!')
            
            else:               
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
                
            cv2.imshow('frame',frame)
        
        else:
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__': 
    # X = webcam_to_data(get_images=True)
    # print(X.shape)
    fix_rectangle()
