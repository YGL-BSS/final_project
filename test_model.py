import new_preprocess as pp
import modeling

import numpy as np
import os

# 학습데이터 불러오기
# dataloader = pp.DataLoader()
# x, y = dataloader.get_xy_all()
# x0, y0 = dataloader.get_xy('000')
# x1, y1 = dataloader.get_xy('001')
# x = np.append(x0, x1, axis=0)
# y = np.append(y0, y1, axis=0)

# print('학습데이터 크기 :', x.shape, y.shape)


# # 모델 만들고 학습하기
# model = modeling.GestureClassification()
# model.put_data(x, y)
# model.start_train()

model = modeling.GestureClassification()
print(model.model.summary())