import modeling
import os
import numpy as np
import cv2

# 데이터 ndarray 만들기
x = np.empty((0, 300, 400, 3))
y = np.empty((0,))

def get_picture_array(img_folder, label):
    data = np.empty((0, 300, 400, 3))
    for picture in os.listdir(img_folder):
        img = cv2.imread(os.path.join(img_folder, picture))
        data = np.append(data, img.reshape((1, 300, 400, 3)), axis=0)
        # print(img.shape, data.shape)
    
    target = np.ones((len(data),)) * label
    return data, target

x0, y0 = get_picture_array('./picture001', 0)
x1, y1 = get_picture_array('./picture001', 1)

x = np.append(x, x0, axis=0)
x = np.append(x, x1, axis=0)
y = np.append(y, y0, axis=0)
y = np.append(y, y1, axis=0)

print('학습데이터 크기 :', x.shape, y.shape)


# 모델 만들고 학습하기
model = modeling.GestureClassification()
model.put_data(x, y)
model.start_train()
