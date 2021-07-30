'''
hand_video 내의 라벨별로 분류된 영상 데이터를 전처리한다.
실행하면 이미지와 이미지 내의 hand box의 좌표를 저장하게 된다.
'''
import config as cf

import pandas as pd
import numpy as np
import wget
import cv2
import os


# Yolo-hand-detection 에서 pretrained 모델 불러오기
if not os.path.exists('./yolo/cross-hands.cfg'):
    wget.download('https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands.cfg', out='./yolo')

if not os.path.exists('./yolo/cross-hands.weights'):
    wget.download('https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands.weights', out='./yolo')

print('Load yolo model...', end='')
net = cv2.dnn.readNet(f'{cf.PATH_YOLO}/cross-hands.weights', f'{cf.PATH_YOLO}/cross-hands.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0]- 1] for i in net.getUnconnectedOutLayers()]
print('Done!\n')


# 영상 데이터 처리할 때 사용하는 함수
def get_coordinate(size, img):
    '''
    Yolo-hand-detection을 활용하여, 손에 해당하는 좌표를 반환한다.
    size : 320, 416, 608 중 택1
    '''
    # frame의 크기 추출
    height, width, _ = img.shape

    # yolo-hand-detection 으로 손 위치 감지
    blob = cv2.dnn.blobFromImage(img, 1/255, (size,size), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # 감지된 좌표 저장
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Hand detect
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                boxes.append([center_x, center_y, w, h])
    
    # 감지된 좌표 하나만 출력
    if len(boxes):
        return boxes[0]
    else:
        return False



# 영상에서 손이 찍힌 원본 이미지 추출하기
labels_path = cf.dir_subdirs_path(cf.PATH_VIDEO)
labels = [os.path.basename(label_path) for label_path in labels_path]

PATH_DATASET = cf.mkdir_under_path(cf.PATH_BASE, 'dataset')
PATH_ORIGIN = cf.mkdir_under_path(PATH_DATASET, 'origin')
PATH_ORIGIN_BOX = cf.mkdir_under_path(PATH_DATASET, 'origin_box')
PATH_COORD = cf.mkdir_under_path(PATH_DATASET, 'coordinate')

# box_coordinates = pd.DataFrame(columns=['label', 'image_name', 'x_start', 'x_end', 'y_start', 'y_end'])
# box_coordinates = pd.DataFrame(columns=['label', 'image_name', 'x', 'y', 'w', 'h']) # x,y는 객체중심좌표, w,h는 객체사이즈


for label, label_path in zip(labels, labels_path):
    print()
    print(f'------ 라벨 저장 : {label}')

    cf.mkdir_under_path(PATH_ORIGIN, label)
    cf.mkdir_under_path(PATH_ORIGIN_BOX, label)
    cf.mkdir_under_path(PATH_COORD, label)

    videos_path = cf.dir_items_path(label_path)
    videos = [os.path.basename(video_path) for video_path in videos_path]

    for video, video_path in zip(videos, videos_path):
        print(f'---- 영상 저장 : {video} ...', end='')
        num_frame = 0
        cam = cv2.VideoCapture(video_path)

        while True:
            success, frame = cam.read()

            success_cnt = 0
            if success:

                # 손 위치 감지하기
                hand_area = get_coordinate(320, frame)
                if hand_area == False:
                    continue
                else:
                    num_frame += 1

                # 5개 frame에 1개씩만 진행하기
                if num_frame % 5 != 1:  # 1, 6, 11, ... 만 저장하기
                    continue
                
                # hand box 조정하기
                center_x, center_y, w, h = hand_area
                x_scale = 1.20
                y_scale = 1.1

                x_start = max(0, int(center_x - w*x_scale/2))
                y_start = max(0, int(center_y - h*y_scale/2))
                x_end = min(frame.shape[1], int(center_x + w*x_scale/2))
                y_end = min(frame.shape[0], int(center_y + h*y_scale/2))

                # hand box 그리기
                frame_box = frame.copy()
                cv2.rectangle(frame_box, (x_start, y_start), (x_end, y_end), (0, 255, 255), 2)

                # Normalized coordinate 계산
                X = (x_end + x_start) / 2 / frame.shape[1]
                Y = (y_end + y_start) / 2 / frame.shape[0]
                W = (x_end - x_start) / frame.shape[1]
                H = (y_end - y_start) / frame.shape[0]

                # 저장하기
                img_name = f'{video[:-4]}_{num_frame:0>5d}'
                cv2.imwrite(
                    f'{PATH_ORIGIN}/{label}/{img_name}.jpg',
                    frame
                )
                cv2.imwrite(
                    f'{PATH_ORIGIN_BOX}/{label}/{img_name}.jpg',
                    frame_box
                )
                # box_coordinate = pd.DataFrame(
                #     [[label, img_name, X, Y, W, H]],
                #     columns=box_coordinates.columns
                # )
                # box_coordinates = box_coordinates.append(box_coordinate)
                f = open(f'{PATH_COORD}/{label}/{img_name}.txt', 'w')
                f.write(f'{labels.index(label)} {X} {Y} {W} {H}')
                f.close()

            else:
                print('Done!')
                break

# # 좌표값 저장하기
# box_coordinates.to_csv(os.path.join(PATH_DATASET, 'coordinates.csv'), index=None)

