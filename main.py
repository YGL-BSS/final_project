'''
메인 실행 코드

실행 할때 아래와 같이 cmd에 입력
python main.py -m train
python main.py -m test
'''
import argparse

import config
import preprocess
import modeling

ap = argparse.ArgumentParser(description='Hand Gesture Password 시스템입니다.')

ap.add_argument('-m', '--mode', required='True', help='[ train / test ] 중 선택')
args = ap.parse_args()

if args.mode == 'train':
    print('Training model...')
    # 학습 코드
    
elif args.mode == 'test':
    print('Testing model...')
    # 테스트 코드
