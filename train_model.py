'''
모델 훈련시키는 코드

순서
1. 이미지를 전처리함
2. 전처리한 훈련 이미지를 가지고 모델을 훈련함
'''

from setting import config

from preprocessing import hand_segmentation
from model import gesture_model

