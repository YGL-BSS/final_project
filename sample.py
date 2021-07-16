from setting import config
import cv2

image_path = config.get_data_path('sample.jpg')

image = cv2.imread(image_path, cv2.IMREAD_COLOR)

print(image.shape)

image_rev = cv2.resize(image, dsize=(240, 240), interpolation=cv2.INTER_AREA)

print(image_rev.shape)

# cv2.imshow('image 1', image)
# cv2.imshow('image rev', image_rev)
# cv2.waitKey()
# cv2.destroyAllWindows()

cv2.imwrite