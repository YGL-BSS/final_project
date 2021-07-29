import config as cf

import pandas as pd
import os

PATH_DATASET = cf.mkdir_under_path(cf.PATH_BASE, 'dataset')

if len(os.listdir(PATH_DATASET)) == 0:
    raise ValueError('video2image.py를 먼저 실행해주세요.')

# 라벨별로 제거 진행
labels_path = cf.dir_subdirs_path('dataset/origin')
labels = [os.path.basename(label_path) for label_path in labels_path]

for label, label_path in zip(labels, labels_path):

    origins_path = cf.dir_items_path(f'dataset/origin/{label}')
    origins = [os.path.basename(origin_path) for origin_path in origins_path]

    origin_boxs_path = cf.dir_items_path(f'dataset/origin_box/{label}')
    origin_boxs = [os.path.basename(origin_box_path) for origin_box_path in origin_boxs_path]

    # 제거해야 할 이미지 파일명 찾기
    del_list = list(set(origins) - set(origin_boxs))

    # origin 폴더에서 제거
    for del_img in del_list:
        print(f'{del_img} 를 data/origin/{label} 에서 제거 중 ...', end='')
        os.remove(f'dataset/origin/{label}/{del_img}')
        print('완료!')

    # coordinates.csv에서 제거
    df = pd.read_csv('dataset/coordinates.csv', index_col=None)
    for del_img in del_list:
        df = df[(df.label == label) & (df.image_name != del_img)]
    df.to_csv('dataset/coordinates.csv', index=None)

print('\n불량 데이터 제거 완료!')

