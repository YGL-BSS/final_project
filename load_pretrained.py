import os
import wget
import zipfile

if not os.path.exists('runs.zip'):
    print('미리 학습된 모델을 불러옵니다.')
    url = 'https://github.com/YGL-BSS/yolov5/releases/download/final/runs.zip'
    wget.download(url, out='./')

if not os.path.isdir('runs'):
    os.mkdir('runs')

print('\n압축 해제 중...', end='')
zipfile.ZipFile('runs.zip').extractall('runs')
print('완료!')