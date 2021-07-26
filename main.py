'''
메인 실행 코드

# ------------ 사용법 ---------------------

옵션이 3가지가 있습니다.
1. [-y] : yolo model을 다운로드하는 옵션
2. [-v] : 영상 데이터에서 손 이미지 데이터로 얻는 옵션 (preprocess.py 사용)
3. [-m] : model을 훈련 및 테스트하는 옵션 (modeling.py 사용)

콘솔창에서의 간단한 실행 예시는 다음과 같습니다.
1. python main.py -y v3         # yolo v3 모델 다운로드
2. python main.py -v live       # webcam에서 손 이미지 얻기
3. python main.py -m train      # 모델 학습 시작

# ----------------------------------------

'''
import argparse
from numpy import save
import wget
import os
from tqdm import tqdm

import config as cf
import new_preprocess as pp
# import modeling

ap = argparse.ArgumentParser(description='Hand Gesture Password 시스템입니다.')

ap.add_argument('-y', '--yolo', default='', help='Yolo-hand-detection 모델 다운로드.')
ap.add_argument('-v', '--video', default='', help='영상 전처리 진행, [ live / recorded ] 중 선택')
ap.add_argument('-m', '--model', default='', help='[ train / test ] 중 선택')
args = ap.parse_args()

cnt = 3     # 수행한 option 수.


# option : yolo
if args.yolo == 'v3':
    url_list = [
        'https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands.cfg',
        'https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands.weights',
    ]
    for url in url_list:
        wget.download(url, out='./yolo')
elif args.yolo == 'v3-prn':
    url_list = [
        'https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands-tiny-prn.cfg',
        'https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands-tiny-prn.weights',
    ]
    for url in url_list:
        wget.download(url, out='./yolo')
elif args.yolo == 'v4':
    url_list = [
        'https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands-yolov4-tiny.cfg',
        'https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands-yolov4-tiny.weights'
    ]
    for url in url_list:
        wget.download(url, out='./yolo')
else:
    cnt -= 1    # 수행한 option 수 감소


# option : video
if args.video == 'live':
    # Webcam video -> dataset
    print('Webcam on!')
    label_name = input('Write the label name >> ')

    detect = pp.HandDetection(selected_model='v3')
    detect.video2hand(save_image_dir=label_name)

elif args.video == 'recorded':
    # Recorded video(.mp4) -> dataset
    print('Recorded video processing!')
    detect = pp.HandDetection(selected_model='v3')

    option = input('Choose option in [ all / part ] >> ')

    # raw_data에 label별로 분류한 모든 폴더의 영상을 전부 손 이미지로 변환하여 저장
    if option == 'all':

        # raw_data 내의 폴더 경로만 불러오기
        labels_path = [os.path.join(cf.PATH_RAW_DATA, f) for f in os.listdir(cf.PATH_RAW_DATA)]
        labels_path = list(filter(os.path.isdir, labels_path))
        labels = [label[-3:] for label in labels_path]
        
        for label, label_path in zip(labels, labels_path):
            videos = os.listdir(label_path)
            videos_path = [os.path.join(label_path, v) for v in videos if v.endswith(r".mp4")]     # .mp4 확장자만 가져옴
            for video_path in videos_path:
                detect.video2hand(video_path=video_path, save_image_dir=label)

        # raw_data_folders = [os.path.join(cf.PATH_RAW_DATA, f) for f in os.listdir(cf.PATH_RAW_DATA)]
        # raw_data_folders_path = list(filter(os.path.isdir, raw_data_folders))
        # for folder, folder_path in zip(raw_data_folders, raw_data_folders_path):
        #     videos = os.listdir(folder_path)
        #     videos_path = [os.path.join(folder_path, v) for v in videos if v.endswith(r".mp4")]

        #     for video_path in videos_path:
        #         detect.video2hand(video_path=video_path, save_image_dir=folder[-3:])
    
    # raw_data에서 특정 폴더의 영상들만 손 이미지로 변환하여 저장
    elif option == 'part':
        label = input('Write the label name >> ')    # ex : label_000를 가져오고 싶으면 000 입력.
        label_path = os.path.join(cf.PATH_RAW_DATA, f'label_{label}')

        if os.path.exists(label_path):
            videos = os.listdir(label_path)
            videos_path = [os.path.join(label_path, v) for v in videos if v.endswith(r".mp4")]

            for video_path in videos_path:
                detect.video2hand(video_path=video_path, save_image_dir=label)

    
    else:
        print('잘못 입력하셨습니다. [ all / part ] 중에서만 입력해주세요.')
        cnt -= 1
else:
    cnt -= 1    # 수행한 option 수 감소


# option : model
if args.model == 'train':
    print('Training model...')
    ##################################
    # 모델 학습 코드
    ##################################
elif args.model == 'test':
    print('Testing model...')
    ##################################
    # 모델 테스트 코드
    ##################################
else:
    cnt -= 1    # 수행한 option 수 감소

if cnt == 0:
    print('Nothing Done, write the code "python main.py -h" for help.')

