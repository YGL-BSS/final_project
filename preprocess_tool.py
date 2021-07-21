# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from hand_detect import get_coordinate, make_model

# paths
now_here = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(now_here, os.pardir))
model_path = os.path.join(base_path, 'model')
data_path = os.path.join(base_path, 'data')
raw_data_path = os.path.join(data_path, 'raw_data')
weights_path = os.path.join(model_path,'cross-hands.weights')
cfg_path = os.path.join(model_path,'cross-hands.cfg')

# material functions

def dir_item(abs_path):
    return [os.path.join(abs_path, item) for item in os.listdir(abs_path)]

def mkdir_under_data(dir_name):
    abs_path = os.path.join(data_path, dir_name)
    try:
        if not os.path.exists(abs_path): os.makedirs(abs_path)
    except OSError: print('can not create dir')

def resize_hand(frame, hand_area, height, width, alpha=0.1):
    x, y, w, h = hand_area
    y_start = round(y * (1-alpha))
    x_start = round(x * (1-alpha))
    y_end = round((y+h) * (1+alpha))
    x_end = round((x+w) * (1+alpha))
    target = frame[y_start:y_end, x_start:x_end]
    return cv2.resize(target, (width, height), interpolation=cv2.INTER_CUBIC)

def write_img(video_name,currentframe,resized_frame):
    folder = os.path.join(data_path, video_name[-11:-8])
    name = f'{video_name[-7:-4]}_{currentframe:0>5}.jpg'
    cv2.imwrite(os.path.join(folder,name), resized_frame)

def onehot_ndarray(y_cnt_lst):
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