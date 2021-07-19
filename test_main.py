from preprocessing import test_segment2 as seg
from preprocessing import read_hand

# seg.cont()

read_hand.get_image_dataset()

x, y = read_hand.get_npy_dataset()

print('x :', x.shape)
print('y :', y.shape)
