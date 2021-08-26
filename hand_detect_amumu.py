import imutils
import numpy as np
import cv2
import os
import hand_detect_functions

# 같은 폴더내에 amumus의 절대 경로 및 자료가 저장된 모션명
amumus = 'amumus'
PATH_BASE = os.path.dirname(__file__)
motions_path = os.path.join(PATH_BASE, amumus)
motions = os.listdir(motions_path)

# 저장할 경로
save_folder = 'saved_amumu'
save_folder_path = os.path.join(PATH_BASE, save_folder)

# 각 데이터 저장 폴더 명 saved_amumu/(folder)
save_datas = ['original', 'mask', 'coordinate']
for data in save_datas:
    try:
        os.mkdir(os.path.join(save_folder_path, data))
    except:
        pass
    # for motion in motions:
    #     os.mkdir(os.path.join(data_path, motion))
original_path = os.path.join(save_folder_path, save_datas[0])
mask_path = os.path.join(save_folder_path, save_datas[1])
coordinate_path = os.path.join(save_folder_path, save_datas[2])
# print(original_path)
# print(mask_path)
# print(coordinate_path)

j = 0
for motion in motions:
    medias_path = os.path.join(motions_path, motion)
    medias = os.listdir(medias_path)
    #print(medias)
    for media in medias:
        media_path = os.path.join(medias_path, media)
        media_name = media.split('.')[0]
        #print(media_path)
        
        frames = cv2.VideoCapture(media_path)

        i = 0
        while True:
            
            grabbed, frame = frames.read()
            if not grabbed: break
            if i % 3 == 0:
                original_name = motion + media_name + f"_{i}" + ".jpg"
                mask_name = motion + media_name + "_masked"+ f"_{i}" + ".jpg"
                coordinate_name = motion + media_name + f"_{i}" + ".txt"
                
                img_height, img_width, _ = frame.shape

                formask = hand_detect_functions.makeMask(frame)
                cv2.imwrite(os.path.join(original_path, original_name), frame)
                cv2.imwrite(os.path.join(mask_path, mask_name), formask)

                x, y, w, h = hand_detect_functions.makeRec(frame)
                motion_label = j
                rel_center_x = (x + (w / 2))/img_width
                rel_center_y = (y + (h / 2))/img_height
                rel_w = w / img_width
                rel_h = h / img_height
                with open(os.path.join(coordinate_path, coordinate_name), "w") as f:
                    f.write(f"{motion_label} {rel_center_x} {rel_center_y} {rel_w} {rel_h}\n")

            i += 1

    j += 1
