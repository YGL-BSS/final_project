'''
도현 형님이 만들어주신 원본 이미지 뻥튀기 코드를 이어받은 코드입니다.
'''

import cv2
import random
import time
import numpy as np

def flip_vertical_img(image):
    '''
    주어진 이미지를 좌우반전하여 반환
    '''
    return cv2.flip(image, 1)

def flip_vertical_coord(coord):
    '''
    주어진 좌표를 좌우반전하여 반환 (x좌표만 바뀜)
    '''
    flip_coord = coord
    flip_coord[1] = str(1 - float(flip_coord[1]))
    return ' '.join(flip_coord)   

def change_brightness(image):
    '''
    주어진 이미지의 밝기를 랜덤으로 변경하여 반환
    '''
    alpha_range = range(2,19)
    return cv2.convertScaleAbs(image, alpha = 0.1 * random.choice(alpha_range)) 

def change_hsv(image):
    '''
    주어진 이미지의 H,S 조절
    '''
    image_hsv = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    h, s, v = cv2.split(image_hsv)
    new_h = ((h.astype('int32') + random.choice(range(5,70))) % 180).astype('uint8')
    new_s = s + random.choice(range(5,20))
    new_img = np.array([new_h, new_s, v]).transpose((1,2,0))
    return cv2.cvtColor(new_img, cv2.COLOR_HSV2BGR)


def resize_shape(image, size=640):
    '''
    정해진 사이즈로 image 변환
    '''
    return cv2.resize(image, dsize=(size, size), interpolation=cv2.INTER_AREA)