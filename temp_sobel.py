import numpy as np
import cv2
import os

import config as cf

# sobel color
for i in range(6):
    folder_path = cf.get_path(cf.PATH_BASE, f'preprocess_dataset\\each25\\{i:0>3d}')
    save_dir = cf.mkdir_under_path(cf.get_path(cf.PATH_BASE, 'preprocess_dataset\\each_sobel25'), f'sobel{i:0>3d}')
    print('save at', save_dir)

    imgs_path = cf.dir_item(folder_path)
    print('Start converting :', folder_path)
    for n, img_path in enumerate(imgs_path):
        frame = cv2.imread(img_path)

        dx3 = cv2.Sobel(frame, cv2.CV_64F, 2, 0, ksize=3)
        dy3 = cv2.Sobel(frame, cv2.CV_64F, 0, 2, ksize=3)

        dx5 = cv2.Sobel(frame, cv2.CV_64F, 2, 0, ksize=5)
        dy5 = cv2.Sobel(frame, cv2.CV_64F, 0, 2, ksize=5)

        sobel3 = cv2.magnitude(dx3, dy3)
        sobel3 = np.clip(sobel3, 0, 255).astype(np.uint8)

        sobel5 = cv2.magnitude(dx5, dy5)
        sobel5 = np.clip(sobel5, 0, 255).astype(np.uint8)

        rate = 0.7
        sobel = cv2.addWeighted(sobel3, 1-rate, sobel5, rate, 0.0)

        save_path = os.path.join(save_dir, f'{i:0>3d}_{n:0>5d}.jpg')
        # print('save ->', save_path)
        cv2.imwrite(save_path, sobel)

    print('End converting :', folder_path)


# # canny
# for i in range(5):
#     folder_path = cf.get_path(cf.PATH_BASE, f'raw_dataset\\origin{i:0>3d}')
#     save_dir = cf.mkdir_under_path(cf.get_path(cf.PATH_BASE, 'preprocess_dataset'), f'canny{i:0>3d}')
#     print('save at', save_dir)

#     imgs_path = cf.dir_item(folder_path)
#     print('Start converting :', folder_path)
#     for n, img_path in enumerate(imgs_path):
#         frame = cv2.imread(img_path)

#         canny = cv2.Canny(frame, 30, 160)

#         save_path = os.path.join(save_dir, f'{i:0>2d}_{n:0>5d}.jpg')
#         # print('save ->', save_path)
#         cv2.imwrite(save_path, canny)

#     print('End converting :', folder_path)



#####################################################


# p = cf.get_path(cf.PATH_BASE, 'raw_dataset/origin000/000_00004.jpg')
# # img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
# img = cv2.imread(p)
# cv2.imshow('origin', img)

# # for i in range(5):
# #     edge_min = 10*(i+1)
# #     canny = cv2.Canny(img, edge_min, 200-edge_min)
# #     cv2.imshow(f'canny{edge_min}', canny)

# # for i in range(5):
# #     dx = cv2.Sobel(img, cv2.CV_64F, 2, 0, ksize=2*i+1)
# #     dy = cv2.Sobel(img, cv2.CV_64F, 0, 2, ksize=2*i+1)

# #     sobel = cv2.magnitude(dx, dy)
# #     sobel = np.clip(sobel, 0, 255).astype(np.uint8)

# #     cv2.imshow(f'sobel{2*i+1}', sobel)

# dx = cv2.Sobel(img, cv2.CV_64F, 2, 0, ksize=3)
# dy = cv2.Sobel(img, cv2.CV_64F, 0, 2, ksize=3)

# sobel1 = cv2.magnitude(dx, dy)
# sobel1 = np.clip(sobel1, 0, 255).astype(np.uint8)

# dx = cv2.Sobel(img, cv2.CV_64F, 2, 0, ksize=5)
# dy = cv2.Sobel(img, cv2.CV_64F, 0, 2, ksize=5)

# sobel2 = cv2.magnitude(dx, dy)
# sobel2 = np.clip(sobel2, 0, 255).astype(np.uint8)

# rate = 0.7
# # sobel = sobel1 * rate + sobel2 * (1 - rate)
# sobel = cv2.addWeighted(sobel1, 1-rate, sobel2, rate, 0.0)
# print(sobel.shape)
# print(type(sobel))
# # print(sobel2[112, 112, :],'->', sobel[112, 112, :])

# cv2.imshow('sobel1', sobel1)
# cv2.imshow('sobel2', sobel2)
# cv2.imshow('sobel', sobel)



# cv2.waitKey()
# cv2.destroyAllWindows()