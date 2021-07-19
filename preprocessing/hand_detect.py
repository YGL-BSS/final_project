import cv2
import numpy as np

def make_model(weights, cfg):
    net = cv2.dnn.readNet(weights, cfg)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0]- 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

def get_coordinate(size, img_path, net, output_layers):
    '''
    size in[320, 416, 608]
    320 as default
    '''
    #path를 받는 경우
    #img = cv2.imread(img_path)
    
    #바로 이미지를 받는 경우
    img = img_path
    
    height, width, _ = img.shape 

    blob = cv2.dnn.blobFromImage(img, 0.00392,(size,size), (0,0,0), True, crop=False)
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

                # Rectangle x_start and y_start
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h]) 
    #print(boxes)
    #print(len(boxes))
    # if len(boxes) == 1:
    #     return boxes[0]
    
    # else:
    #     return 'there is no hand or too many hands'
    if len(boxes):
        return boxes[0]
    else:
        return False

# input 이미지의 형태를 변형
# size가 커질 수록 accuracy up, speed down
# size = [320, 416, 608]

# 이미지 경로(추후 영상에서 프레임 단위로 받아서 처리 가능할 듯)
# img_path = 'your path'

# 모델 정보
# weights = 'your path'
# cfg = 'your path'

# 손을 감지하고 손을 포함한 직사각형 영역을 반환 후 출력
if __name__ == '__main__':
    coordinate = get_coordinate(size[0], img_path, weights, cfg)
    print(coordinate)
