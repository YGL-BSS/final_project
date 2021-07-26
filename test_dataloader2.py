'''
preprocess_data에 있는 이미지 파일들을 numpy로 불러오는 코드
'''
import config as cf     # 우리가 만든 config.py를 import함

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import os
import cv2


class DataLoader():
    '''
    preprocess_data 디렉토리에 저장된 레이블 별 사진 데이터를
    모델에 학습시킬 데이터로 가져와 처리하는 class입니다.

    
    '''
    def __init__(self):
        self.folders = []           # 라벨링된 이미지를 담은 폴더 명 (라벨명)
        self.folders_path = []      # 라벨링된 이미지를 담은 폴더 경로
        for folder in os.listdir(cf.PATH_PREPROCESS_DATA):
            folder_path = os.path.join(cf.PATH_PREPROCESS_DATA, folder)
            if os.path.isdir(folder_path):
                self.folders.append(folder)
                self.folders_path.append(folder_path)
        
        # 라벨명 인코딩 : 폴더명(라벨명) -> 정수(int) -> 원핫인코딩
        # 링크 참고
        # https://john-analyst.medium.com/%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A0%84%EC%B2%98%EB%A6%AC-%EB%A0%88%EC%9D%B4%EB%B8%94-%EC%9D%B8%EC%BD%94%EB%94%A9%EA%B3%BC-%EC%9B%90%ED%95%AB-%EC%9D%B8%EC%BD%94%EB%94%A9-f0220df21df1
        self.encoder_label = LabelEncoder()     # 인코더 : 폴더명(라벨명) -> 정수(int)
        self.encoder_onehot = OneHotEncoder()   # 인코더 : 정수(int) -> 원핫인코딩
        self.labels = self.encoder_label.fit_transform(self.folders)
        self.onehots = self.encoder_onehot.fit_transform(self.labels.reshape(-1, 1)).toarray()

        self.X = None
        self.Y = None

    def get_xy_all(self):
        '''
        preprocess_data의 라벨링에 따라 데이터셋을 만든다.
        X, Y를 반환한다.
        '''
        # 반환할 X, Y 초기화
        X = None
        Y = None

        # 각 label 폴더의 이미지들을 모두 조회
        for folder, folder_path in zip(self.folders, self.folders_path):
            print(f'Loading label : {folder}')
            image_num = 0
            for image_path in [os.path.join(folder_path, img) for img in os.listdir(folder_path)]:
                image = cv2.imread(image_path)
                image = np.expand_dims(image, axis=0)
                if X is None:
                    X = image
                    
                else:
                    X = np.append(X, image, axis=0)
                
                image_num += 1
            
            y_temp = [folder] * image_num
            y_temp = self.encoder_label.transform(y_temp)
            y_temp = self.encoder_onehot.transform(y_temp.reshape(-1, 1)).toarray()
            if Y is None:
                Y = y_temp
            else:
                Y = np.append(Y, y_temp, axis=0)
        
        # 데이터셋을 클래스 변수에 저장
        self.X, self.Y = X, Y

        return X, Y

    
    def get_xy(self, label_name):
        '''
        preprocess_data에서 원하는 라벨 이미지들만 데이터로 가져온다.(get_xy_all의 하위호환)
        X, Y를 반환한다. 
        '''
        # 반환할 X, Y 초기화
        X = None
        Y = None

        # 해당 라벨명이 preprocess_data에 존재하는지 여부 확인
        if label_name in self.folders:
            print(f'Loading label : {label_name}')
            image_num = 0
            label_path = os.path.join(cf.PATH_PREPROCESS_DATA, label_name)

            # 폴더 내의 모든 이미지를 하나씩 불러와서 ndarray로 저장
            for image_path in [os.path.join(label_path, img) for img in os.listdir(label_path)]:
                image = cv2.imread(image_path)
                image = np.expand_dims(image, axis=0)
                if X is None:
                    X = image
                    
                else:
                    X = np.append(X, image, axis=0)
                
                image_num += 1
            
            y_temp = [label_name] * image_num
            y_temp = self.encoder_label.transform(y_temp)
            y_temp = self.encoder_onehot.transform(y_temp.reshape(-1, 1)).toarray()
            if Y is None:
                Y = y_temp
            else:
                Y = np.append(Y, y_temp, axis=0)

            return X, Y

        else:
            print('There is no such label in "preprocess_data" directory.')
            return False



if __name__ == '__main__': 

    dataloader = DataLoader()
    x, y = dataloader.get_xy_all()
    x_000, y_000 = dataloader.get_xy('label_000')

    print(x.shape, y.shape)
    print(x_000.shape, y_000.shape)