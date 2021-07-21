# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2

# paths
now_here = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(now_here, os.pardir))
model_path = os.path.join(base_path, 'model')
data_path = os.path.join(base_path, 'data')
raw_data_path = os.path.join(data_path, 'raw_data')
weights_path = os.path.join(model_path,'cross-hands.weights')
cfg_path = os.path.join(model_path,'cross-hands.cfg')

# material functions

def make_model(weights, cfg):
    net = cv2.dnn.readNet(weights, cfg)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0]- 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

def get_coordinate(size, img_path, net, output_layers):
    '''
    size in [320, 416, 608]
    choose 320 for standard

    img_path directly takes image(frame,ndarray)
    not 'PATH'
    '''
    #path를 받는 경우
    #img = cv2.imread(img_path)
    
    #바로 이미지를 받는 경우
    img = img_path
    
    height, width, _ = img.shape 

    blob = cv2.dnn.blobFromImage(img, 0.00392,(size,size), (0,0,0), True, crop=False)
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
    #print(boxes)
    #print(len(boxes))
    # if len(boxes) == 1:
    #     return boxes[0]
    
    # else:
    #     return 'there is no hand or too many hands'
    if len(boxes):
        return boxes[0]
    else:
        return False



def dir_item(abs_path):
    '''
    매개변수로 특정 디렉토리의 절대경로를 받는다.
    해당 디렉토리 내의 모든 아이템들의 절대경로를 
    리스트에 담아 반환한다.
    '''
    return [os.path.join(abs_path, item) for item in os.listdir(abs_path)]

def mkdir_under_data(dir_name):
    '''
    디폴트로 설정된 디렉토리에
    dir_name을 폴더명으로 갖는
    하위 디렉토리를 생성한다.
    '''
    abs_path = os.path.join(data_path, dir_name)
    try:
        if not os.path.exists(abs_path): os.makedirs(abs_path)
    except OSError: print('can not create dir')

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

def write_img(video_name,currentframe,resized_frame):
    '''
    설정된 경로에서 비디오의 앞 3글자 디렉토리에
    비디오명 끝 3자리(.확장자명 제외)를 앞 이름을 갖는
    이미지를 저장한다.
    '''
    folder = os.path.join(data_path, video_name[-11:-8])
    name = f'{video_name[-7:-4]}_{currentframe:0>5}.jpg'
    cv2.imwrite(os.path.join(folder,name), resized_frame)

def onehot_ndarray(y_cnt_lst):
    '''
    예시) [333,222,550]이라는 리스트를 매개변수로 전달할 경우
    위에서부터 차례로
    [1 0 0] 이 333개
    [0 1 0] 이 222개
    [0 0 1] 이 550개
    append된 2차원 ndarray를 반환. 이경우 shape는 (1105,3)
    '''
    Y = np.empty((0,len(y_cnt_lst)),dtype=int)
    for i in range(len(y_cnt_lst)):
            temp = [0] * len(y_cnt_lst)
            temp[i] = 1
            for j in range(y_cnt_lst[i]):
                y_temp = np.reshape(temp,(1, len(y_cnt_lst)))
                Y = np.append(Y,y_temp,axis=0)
    return Y

# main function

def videos_to_data(height=224, width=224, channel=3, get_npy=True, get_images=False):
    '''
    get_npy=True 기본값
    기본적으로 (X_data, Y_data)를 반환. 각각 ndarray
    그와 동시에 npy 파일을 저장

    get_images=False
    True로 설정 시 이미지를 각label 디렉토리에 저장
    height, width 기본값 224
    '''
    
    
    
    #model set
    net, output_layers = make_model(weights_path,cfg_path)

    #create ndarray as default
    if get_npy:
        data_array = np.empty((0,height,width,channel))
        y_cnt_lst = []
        y_cnt_temp = 0
    
    #loop for each label
    for label_path in dir_item(raw_data_path):
        
        #create folder for images when 'get_images=True'
        if get_images: mkdir_under_data(label_path[-3:])

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
                    try:
                        resized_frame = resize_hand(frame,hand_area,height,width)
                    except:
                        continue

                    #append data at ndarray as default
                    if get_npy:
                        data_array = np.append(
                            data_array, resized_frame.reshape((1,height,width,channel)), axis=0
                        )
                    
                    #write image when 'get_images=True'
                    if get_images:
                        write_img(video_path,currentframe,resized_frame)
                    currentframe += 1
                    y_cnt_temp += 1
                else: break
            cam.release()
            cv2.destroyAllWindows()
        
        if get_npy:
            y_cnt_lst.append(y_cnt_temp)
            y_cnt_temp = 0
    if get_npy:
        np.save(f'{os.path.join(data_path,"X_data")}',data_array)
        Y = onehot_ndarray(y_cnt_lst)
        np.save(f'{os.path.join(data_path,"Y_data")}',Y)
        return data_array, Y


if __name__ == '__main__': 
    X,Y = videos_to_data(get_images=True)
    print(X.shape)
    print(Y.shape)