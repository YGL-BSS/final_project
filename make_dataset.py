'''
원본 이미지, 좌표 txt 파일들에 augmentation, split을 적용하여
데이터셋으로 만드는 코드입니다.

# 이미지 이름 규칙
    - 좌우 반전된 이미지는 "_flip"이 추가됨
    - 밝기, 색상 및 채도가 변경된 이미지는 "_trans"가 추가됨

Usage:
    $ python make_dataset.py --dataset real20 --multi-num 3 --name sample1
    $ python make_dataset.py -d real20 -m 3 -n sample1
'''

import argparse
from pathlib import Path
import shutil
import sys
import os

import cv2
import random
import time
from tqdm import tqdm

from utils.custom_datasets import flip_vertical_img, flip_vertical_coord, change_brightness, change_hsv, resize_shape

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add this directory to path
DATASETS = FILE.parents[0] / 'datasets'
DATA = FILE.parents[0] / 'data'


# print(FILE)
# print(DATASETS)

def split_data(dir_target, dir_save, split_rate=(0.8, 0.1, 0.1), flip=False):
    '''
    dir_target 디렉토리 안의 images, labels 안의 파일들을
    train, valid, test로 나누어서 dir_save 디렉토리에 저장하는 코드
    '''
    # dir_target의 이미지 이름 리스트 추출
    images = dir_target / 'images'
    labels = dir_target / 'labels'
    names = [n.stem for n in images.iterdir()]

    # 이미지 이름 리스트를 랜덤으로 나누기
    rnd_idx = [n for n in range(len(names))]
    random.shuffle(rnd_idx)

    split_idx = \
        [0, round(len(names)*sum(split_rate[:1])), round(len(names)*sum(split_rate[:2])), len(names)]
    names_splits = \
        [names[split_idx[i]:split_idx[i+1]] for i in range(3)]      # [names_train, names_valid, names_test]

    # 나누어진 이미지 이름 리스트에 따라 dir_save에 하나씩 옮겨 저장하기
    folders_split = ['train', 'valid', 'test']
    for folder_split, names_split in zip(folders_split, names_splits):
        # create directory
        if not (dir_save / folder_split).exists():
            (dir_save / folder_split).mkdir()
        i = 0
        for name in tqdm(names_split, desc=f'split [{folder_split}]'):
            i += 1
            if i % 4 != 0:  # 데이터 수를 1/4배.
                continue
            new_images = dir_save / folder_split / 'images'
            new_labels = dir_save / folder_split / 'labels'
            if not new_images.exists(): new_images.mkdir()
            if not new_labels.exists(): new_labels.mkdir()
            # shutil.copy(images / f'{name}.jpg', new_images / f'{name}.jpg')
            img = cv2.imread(str(images / f'{name}.jpg'))
            if flip and (i%2 == 1):
                img = flip_vertical_img(img)                                        # 홀수번째 사진만 좌우 flip
                cv2.imwrite(str(new_images / f'{name}.jpg'), resize_shape(img))     # 640x640으로 변환 후 저장
            else:
                cv2.imwrite(str(new_images / f'{name}.jpg'), resize_shape(img))     # 640x640으로 변환 후 저장
            shutil.copy(labels / f'{name}.txt', new_labels / f'{name}.txt')


def add_vertical_flip(dir_target):
    '''
    dir_target 디렉토리 안의 images, labels 안의 이미지 파일들을
    좌우 반전시켜서 원본 이미지를 2배로 늘리는 코드
    '''
    images = dir_target / 'images'
    labels = dir_target / 'labels'
    names = [n.stem for n in images.iterdir()]

    for name in tqdm(names, desc=f'flipping [{dir_target.name}]'):

        # 기존 이미지, 좌표 불러오기
        img = cv2.imread(str(images / f'{name}.jpg'))
        with open(labels / f'{name}.txt', 'r') as f:
            coords = f.read()
        coords = coords.split('\n')     # str -> list

        # 좌우 반전 이미지 생성
        new_img_flip = img.copy()
        new_img_flip = flip_vertical_img(new_img_flip)
        # 좌우 반전 좌표 생성
        new_coords_flip = coords
        for i, coord in enumerate(coords):
            try:
                coord = coord.split()
                new_coords_flip[i] = flip_vertical_coord(coord)
            except:
                break
        
        # 좌우 반전 이미지 저장
        # cv2.imwrite(str(images / f'{name}_flip.jpg'), new_img_flip)
        cv2.imwrite(str(images / f'{name}_flip.jpg'), resize_shape(new_img_flip))   # 640x640으로 변환 후 저장
        # 좌우 반전 좌표 저장
        with open(labels / f'{name}_flip.txt', 'w') as f:
                new_coords_flip = '\n'.join(new_coords_flip)
                f.write(new_coords_flip)


def pop_rice(dir_target, multi_num):
    '''
    dir_target 디렉토리 안의 images, labels 안의 이미지 파일들을
    밝기 조절, 색상+채도 조절해서 원본 이미지를 뻥튀기하는 코드
    '''
    images = dir_target / 'images'
    labels = dir_target / 'labels'
    names = [n.stem for n in images.iterdir()]

    # 밝기 조절 여부, 색상+채도 조절 여부
    trans = [[False,True], [True,False], [True,True]]

    for name in tqdm(names, desc=f'pop rice [{dir_target.name}]'):

        img = cv2.imread(str(images / f'{name}.jpg'))
        with open(labels / f'{name}.txt', 'r') as f:
            coords = f.read()
        coords = coords.split('\n')

        # 원하는 개수만큼 augmentation하기
        for i in range(multi_num):
            new_img = img.copy()
            new_coords= coords
            is_bright, is_hsv = random.choice(trans)

            if is_bright:
                new_img = change_brightness(new_img)
            if is_hsv:
                new_img = change_hsv(new_img)
            
            # 새로운 이미지 저장
            # cv2.imwrite(str(images / f'{name}_trans{i}.jpg'), new_img)
            cv2.imwrite(str(images / f'{name}_trans{i}.jpg'), resize_shape(new_img))    # 640x640으로 변환 후 저장
            # 새로운 좌표 저장
            with open(labels / f'{name}_trans{i}.txt', 'w') as f:
                new_coords = '\n'.join(new_coords)
                f.write(new_coords)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, help='name of your_dataset where ./datasets/(your_dataset)')
    parser.add_argument('--multi-num', '-m', type=int, default=3, help='this is the number you want to increase')
    parser.add_argument('--name', '-n', type=str, default='new_dataset', help='name for new dataset')
    
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt):
    
    # 원본 데이터셋 위치 지정
    try:
        DATASET = DATASETS / opt.dataset
    except:
        raise ValueError('please write dataset name only in ./datasets directory')
    
    # 처리완료된 데이터셋 위치 지정
    NEWDATA = DATA / opt.name
    if not NEWDATA.exists():
        NEWDATA.mkdir()
    else:
        # 이미 존재하면 삭제후 다시 생성
        answer = input('기존 데이터셋을 삭제하시겠습니까? [y/n] : ')
        if answer in ['y', 'Y']:
            shutil.rmtree(NEWDATA)  # NEWDATA.rmdir()
            NEWDATA.mkdir()
        elif answer in ['n', 'N']:
            raise ValueError('no를 선택하셨습니다. 기존 데이터셋을 백업하고 다시 실행해주세요.')
        else:
            raise ValueError('yes와 no 중에서만 입력해주십시오.')
    
    # train, valid, test 로 나누기
    print('split 시작')
    split_data(DATASET, NEWDATA, flip=True)
    print('완료!')

    # train, valid, test 폴더의 이미지들을 좌우반전하여 추가하기
    # print('좌우 반전 추가 시작')
    # add_vertical_flip(NEWDATA / 'train')
    # add_vertical_flip(NEWDATA / 'valid')
    # add_vertical_flip(NEWDATA / 'test')
    # print('좌우 반전 추가 완료!\n')

    # train, valid, test 폴더의 이미지들을 각각 augmentation으로 뻥튀기하기
    print(opt.multi_num, '배로 뻥튀기 시작')
    pop_rice(NEWDATA / 'train', opt.multi_num)
    pop_rice(NEWDATA / 'valid', opt.multi_num)
    pop_rice(NEWDATA / 'test', opt.multi_num)
    print('완료!')

    # data.yaml 파일 만들기
    with open(NEWDATA / 'data.yaml', 'w') as f:
        f.write(f'train: ./data/{opt.name}/train\n')
        f.write(f'val: ./data/{opt.name}/valid\n\n')
        f.write(f'nc: 8\n')
        f.write(f"names: ['0', '1', '2', '3', '4', '5', '6', '7']")

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

