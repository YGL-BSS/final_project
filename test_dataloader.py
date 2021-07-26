import new_preprocess as pp
import numpy as np




dataloader = pp.DataLoader()
x, y = dataloader.get_xy_all()

print(x.shape, y.shape)
