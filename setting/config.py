import os

here = os.path.dirname(__file__)

PATH_BASE = os.path.abspath(os.path.join(here, os.pardir))
PATH_DATA = os.path.join(PATH_BASE, 'data')

# print(here)
# print(PATH_BASE)

def get_data_path(data_name):
    '''
    data 폴더의 데이터 파일의 경로를 반환한다.
    '''
    return os.path.join(PATH_DATA, data_name)
