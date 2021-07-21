'''
전역 변수 설정
'''

import os

PATH_BASE = os.path.dirname(__file__)
PATH_PREPROCESS_DATA = os.path.join(PATH_BASE, 'preprocess_data')
PATH_RAW_DATA = os.path.join(PATH_BASE, 'raw_data')
PATH_SAVED_MODEL = os.path.join(PATH_BASE, 'saved_model')
PATH_TEST_VIDEO = os.path.join(PATH_BASE, 'test_video')
PATH_YOLO = os.path.join(PATH_BASE, 'yolo')

def get_path(PATH, data_name):
    '''
    PATH폴더의 데이터 파일의 경로를 반환한다.
    '''
    return os.path.join(PATH, data_name)

def dir_item(abs_path):
    '''
    매개변수로 특정 디렉토리의 절대경로를 받는다.
    해당 디렉토리 내의 모든 아이템들의 절대경로를 
    리스트에 담아 반환한다.
    '''
    return [os.path.join(abs_path, item) for item in os.listdir(abs_path)]

def mkdir_under_path(PATH, dir_name):
    '''
    PATH 디렉토리에(절대경로)
    dir_name을 폴더명으로 갖는
    하위 디렉토리를 생성한다.
    생성된 하위디렉토리의 절대경로를 반환
    '''
    abs_path = os.path.join(PATH, dir_name)
    try:
        if not os.path.exists(abs_path):
            os.makedirs(abs_path)
            return abs_path
    except OSError:
        print('can not create dir')



if __name__ == '__main__':
    print('base path :', PATH_BASE)
    print('preprocess data path :', PATH_PREPROCESS_DATA)
    print('raw data path :', PATH_RAW_DATA)
    print('saved model path :', PATH_SAVED_MODEL)
    print('test video path :', PATH_TEST_VIDEO)
    print('yolo path :', PATH_YOLO)