'''
훈련 시킨 모델을 검증하는 코드

# 순서
    1. 검증하고자 하는 원본 이미지 불러옴
    2. 원본 이미지에서 손 이미지 추출
    3. 학습된 모델에 손 이미지 입력해서, 예측한 라벨 출력
'''

from setting import config

from preprocessing.hand_detect import get_coordinate, make_model
from model import gesture_model as gm

from tensorflow.keras.models import load_model

import numpy as np
import cv2
import os

# load image
path_img = config.get_data_path('sample2.jpg')
img = cv2.imread(path_img)

# 원본 이미지 -> 손 이미지
weights = os.path.join(config.PATH_MODEL,'cross-hands.weights')
cfg = os.path.join(config.PATH_MODEL,'cross-hands.cfg')
net, output_layers = make_model(weights,cfg)

image_path = config.get_data_path('sample.jpg')
frame = cv2.imread(image_path, cv2.IMREAD_COLOR)    # numpy

height, width, channel = 300, 400, 3
data_array = np.empty((0,height,width,channel))
y_lst = []
y_temp = 0

hand_area = get_coordinate(320, frame, net, output_layers)

#when hand area not detected
if not hand_area:
    print(hand_area)

else:
#slice hand area and resize
    hand_x, hand_y, w, h = hand_area
    hand_slice = frame[round(hand_y*0.9):round((hand_y+h)*1.1), round(hand_x*0.9):round((hand_x+w)*1.1)]
    resized_frame = cv2.resize(hand_slice, (width, height), interpolation=cv2.INTER_CUBIC)


    # write image when 'get_images=True'

    cv2.imwrite(config.get_data_path('sample_hand.jpg'), resized_frame)

# x_pred = resized_frame
x_pred = cv2.imread(config.get_data_path('sample3.jpg'), cv2.IMREAD_COLOR)
print(x_pred.shape)

x_pred = np.reshape(x_pred, (1, 300, 400, 3))
print(x_pred.shape)

# load model
path_model = config.get_model_path('trained_model/gesture_model')
# model = load_model(path_model)
model = gm.create_model()
model.load_weights(path_model)

# predict labeling
y_pred = model.predict(x_pred)

print('\n')
print('예측 결과 :', y_pred)
print('\n')