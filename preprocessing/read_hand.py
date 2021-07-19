'''
손 이미지 불러오는 코드
'''

from setting import config

import numpy as np
import os

def get_image_dataset():

    # 손 이미지 파일 경로
    PATH_HAND = config.get_data_path('hand_data')

    # 라벨링 된 폴더별로 데이터셋 만들기
    for label in os.listdir(PATH_HAND):
        
        PATH_LABEL = os.path.join(PATH_HAND, label)

        print(len(os.listdir(PATH_LABEL)))

def get_npy_dataset():

    x = np.load(config.get_data_path('X_data.npy'))
    y = np.load(config.get_data_path('Y_data.npy'))

    return x, y



