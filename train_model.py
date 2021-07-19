'''
모델 훈련시키는 코드

# 순서
    1. 영상을 원본 이미지로 변환
    2. 원본 이미지에서 손 이미지 추출
    3. 손 이미지를 라벨링하는 모델 훈련
'''

from setting import config                      # 디렉토리 불러오는 모듈

from videos_to_datasets import prepocess_data   # 영상 -> 원본 이미지
from preprocessing import hand_detect           # 원본 이미지 -> 손 이미지
from model import gesture_model as gm           # 손 이미지 -> 라벨링

# preprocess
prepocess_data(width=400, height=300, get_images=True)  # 영상 -> 원본 이미지
pass                                                    # 원본 이미지 -> 손 이미지
pass                                                    # 손 이미지 -> resize

# train model
epoch = 50
batch = 64

model = gm.create_model()
model.fit(
    x_train, y_train, epochs=epoch, batch_size=batch,
    validation_data=(x_valid, y_valid),
    callbacks=gm.create_callback,
    verbose=1,
)

print('\n')
print('Model Train Done')
print('\n')

