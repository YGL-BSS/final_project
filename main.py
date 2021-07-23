'''
메인 실행 코드

실행 할때 아래와 같이 cmd에 입력
python main.py -m train
python main.py -m test
'''
import argparse
import wget
import os

# import config
# import preprocess
# import modeling

ap = argparse.ArgumentParser(description='Hand Gesture Password 시스템입니다.')

ap.add_argument('-m', '--mode', required='True', default='', help='[ ready / train / test ] 중 선택')
args = ap.parse_args()

if args.mode == 'ready':
    print('Ready for Survice...')
    
    # Yolo pretrained model 불러오기
    if not os.path.isdir('./yolo'):
        os.mkdir('./yolo')
    
    url_list = [
        'https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands.cfg',
        'https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands.weights',

        # 'https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands-tiny-prn.cfg',
        # 'https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands-tiny-prn.weights',
        
        # 'https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands-yolov4-tiny.cfg',
        # 'https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands-yolov4-tiny.weights'
    ]

    for url in url_list:
        wget.download(url, out='./yolo')

elif args.mode == 'train':
    print('Training model...')
    # 학습 코드
    
elif args.mode == 'test':
    print('Testing model...')
    # 테스트 코드

else:
    print('')
