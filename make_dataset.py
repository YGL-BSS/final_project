'''
refine_dataset.py 돌린 이후에, 데이터셋으로 정리하는 코드
'''

import config as cf

import pandas as pd
import os
import shutil
from tqdm import tqdm

PATH_DATASET = cf.mkdir_under_path(cf.PATH_BASE, 'dataset')

if len(os.listdir(PATH_DATASET)) == 0:
    raise ValueError('video2image.py를 먼저 실행해주세요.')

PATH_ORIGIN = cf.mkdir_under_path(PATH_DATASET, 'origin')
PATH_COORD = cf.mkdir_under_path(PATH_DATASET, 'coordinate')

PATH_DATA = cf.mkdir_under_path(PATH_DATASET, 'data')
PATH_IMG = cf.mkdir_under_path(PATH_DATA, 'img')
PATH_BACKUP = cf.mkdir_under_path(PATH_DATA, 'backup')

# 라벨별로 이미지 파일, box 좌표 파일 가져오기
labels = [os.path.basename(label_path) for label_path in cf.dir_subdirs_path(PATH_ORIGIN)]
for label in labels:
    print('현재 처리중인 라벨 :', label)
    origins_path = cf.dir_items_path(os.path.join(PATH_ORIGIN, label))  # dataset/origin/(label)/*.jpg
    coords_path = cf.dir_items_path(os.path.join(PATH_COORD, label))    # dataset/coorindate/(label)/*.jpg
    if len(origins_path) != len(coords_path):
        raise ValueError('origin과 coorinate 내의 파일 개수가 다릅니다.')
    
    data_names = [os.path.basename(file_name)[:-4] for file_name in origins_path]
    for data_name, origin_path, coord_path in tqdm(zip(data_names, origins_path, coords_path)):
        new_data_name = f'{label}-{data_name}'
        shutil.copy(origin_path, os.path.join(PATH_IMG, f'{new_data_name}.jpg'))
        shutil.copy(coord_path, os.path.join(PATH_IMG, f'{new_data_name}.txt'))


# obj.data 파일 만들기
f = open(f'{PATH_DATA}/obj.data', 'w')
f.write('classes = {}\n'.format(len(labels)))
f.write('train = {}\n'.format('data/train.txt'))
f.write('valid = {}\n'.format('data/train.txt'))  # valid.txt 생성해야하나?
f.write('names = {}\n'.format('data/obj.names'))
f.write('backup = {}'.format('data/backup/'))
f.close()

# obj.names 파일 만들기
f = open(f'{PATH_DATA}/obj.names', 'w')
for label in labels:
    f.write(label)
    f.write('\n')
f.close()

# train.txt 빈 파일 만들기
f = open(f'{PATH_DATA}/train.txt', 'w')
f.close()

# valid.txt 빈 파일 만들기
f = open(f'{PATH_DATA}/valid.txt', 'w')
f.close()
