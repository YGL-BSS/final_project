'''
메인 실행 코드
'''

import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-m', '-mode', default='train', help='train / test')
args = ap.parse_args()

if args.network == 'test':
    print('Testing model...')
    # 테스트 코드
else:
    print('Training model...')
    # 학습 코드