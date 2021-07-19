'''
모델 훈련시키는 코드

# 순서
    1. 영상을 원본 이미지로 변환
    2. 원본 이미지에서 손 이미지 추출
    3. 손 이미지를 라벨링하는 모델 훈련
'''

from setting import config                      # 디렉토리 불러오는 모듈

from videos_to_datasets import prepocess_data as v2i    # 영상 -> 원본 이미지
from now_on_test import prepocess_data as i2h           # 원본 이미지 -> 손 이미지

from preprocessing import read_hand as rh       # 손 이미지 불러오기
from model import gesture_model as gm           # 손 이미지 -> 라벨링

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.model_selection import train_test_split


# preprocess
# v2i(width=400, height=300, get_images=True)  # 영상 -> 원본 이미지
# i2h(width=400,height=300,get_images=True)    # 원본 이미지 -> 손 이미지

# 전처리된 손 이미지 불러오기
x, y = rh.get_npy_dataset()
x = x / 255
y = y / 255

# 훈련 데이터, 검증 데이터 분리하기
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, shuffle=True)


# train model
epoch = 50
batch = 64

model = gm.create_model(num_label=2)

model.fit(
    x_train, y_train, epochs=epoch, batch_size=batch,
    validation_data=(x_valid, y_valid),
    callbacks=gm.create_callback(),
    verbose=1,
)

print('\n')
print('Model Train Done')
print('\n')

