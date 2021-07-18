import cv2
import numpy as np

def get_coordinate(size,img_path,weights, cfg):
    
    net = cv2.dnn.readNet(weights, cfg)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0]- 1] for i in net.getUnconnectedOutLayers()]

    img = cv2.imread(img_path)
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
    if len(boxes) == 1:
        return boxes[0]
    
    else:
        return 'there is no hand or too many hands'

size = [320, 416, 608]
img_path = 'C:/Python/YGL_final_project/Pictures/20210716_145851.jpg'
weights = 'C:/Python/YGL_final_project/hand-detection-test/cross-hands.weights'
cfg = 'C:/Python/YGL_final_project/hand-detection-test/cross-hands.cfg'

coordinate = get_coordinate(size[0], img_path, weights, cfg)
print(coordinate)