'''
훈련 시킨 모델을 검증하는 코드

# 순서
    1. 검증하고자 하는 원본 이미지 불러옴
    2. 원본 이미지에서 손 이미지 추출
    3. 학습된 모델에 손 이미지 입력해서, 예측한 라벨 출력
'''

from setting import config
import cv2

from preprocessing import hand_detect
from model import gesture_model as gm

from tensorflow.keras.models import load_model

# load image
path_img = config.get_data_path('sample.jpg')
img = cv2.imread(path_img)

# preprocess
pass                                                    # 원본 이미지 -> 손 이미지
pass                                                    # 손 이미지 -> resize

# load model
path_model = config.get_model_path('gesture_model.ckpt')
model = load_model(path_model)

# predict labeling
y_pred = model.predict(x_pred)

print('\n')
print('예측 결과 :', y_pred)
print('\n')