import modeling

import numpy as np


# 모델 만들고 학습하기
model = modeling.GestureClassification()
print(model.model.summary())
