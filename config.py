'''
전역 변수 설정
'''

import os

PATH_BASE = os.path.dirname(__file__)
PATH_DATA = os.path.join(PATH_BASE, 'data')
PATH_MODEL = os.path.join(PATH_BASE, 'model')
PATH_LOG = os.path.join(PATH_BASE, 'log')

def get_data_path(data_name):
    '''
    data 폴더의 데이터 파일의 경로를 반환한다.
    '''
    return os.path.join(PATH_DATA, data_name)

def get_model_path(model_name):
    '''
    model 폴더의 모델 저장 파일의 경로를 반환한다.
    '''
    return os.path.join(PATH_MODEL, model_name)

def get_log_path(log_name):
    '''
    log 폴더의 로그 파일 경로 지정
    '''
    return os.path.join(PATH_LOG, log_name)



if __name__ == '__main__':
    print('base path :', PATH_BASE)
    print('base path :', PATH_DATA)
    print('base path :', PATH_MODEL)
    print('base path :', PATH_LOG)