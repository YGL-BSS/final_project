import config as cf

import numpy as np
import time
import cv2
import os

import colorsys
import random

model_name = 'custom-yolov4-tiny-detector'
target_len = 608

weights = os.path.join(cf.PATH_YOLO, f'{model_name}_final.weights')
cfg = os.path.join(cf.PATH_YOLO, f'{model_name}.cfg')
if not os.path.exists(weights):
    raise KeyError('There is no weights file')
if not os.path.exists(cfg):
    raise KeyError('There is no cfg file')

net = cv2.dnn.readNet(weights, cfg)
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

labels = []
with open(os.path.join(cf.PATH_YOLO, 'obj.names'), 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# 함수 정의
def image_preprocess(image):
    # target_len = 416
    image_h, image_w, _ = image.shape

    scale = min(target_len/image_w, target_len/image_h)
    re_w, re_h = int(scale*image_w), int(scale*image_h)
    image_resized = cv2.resize(image, (re_w, re_h))

    image_padded = np.full(shape=[target_len, target_len, 3], fill_value=128.0) # 회색 정사각형 이미지
    pad_w, pad_h = (target_len - re_w) // 2, (target_len - re_h) // 2           # 회색으로 남을 영역 크기
    image_padded[pad_h:pad_h+re_h, pad_w:pad_w+re_w, :] = image_resized

    image_padded = image_padded.astype(np.uint8)

    return image_padded

def get_bboxes(image, net, output_layers):
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 1/255, (target_len, target_len), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # print('outs :', type(outs), len(outs[0]), len(outs[1]), len(outs[2]))

    bboxes = np.concatenate(outs)
    bboxes[:, 0] = bboxes[:, 0] * width
    bboxes[:, 1] = bboxes[:, 1] * height
    bboxes[:, 2] = bboxes[:, 2] * width
    bboxes[:, 3] = bboxes[:, 3] * height

    return bboxes

def draw_bbox(image, bboxes, show_label=True, show_confidence=True, Text_colors=(0, 0, 0)):
    num_labels = len(labels)
    height, width, _ = image.shape

    # hsv --> rgb로 mapping
    hsv_tuples = [(1.0 * x / num_labels, 1., 1.) for x in range(num_labels)]
    # print('hsv_tuples', hsv_tuples)
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0]*255), int(x[1]*255), int(x[2]*255)), colors))
    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    if len(bboxes) == 0:
        return image
    # print(bboxes[0])
    for i, bbox in enumerate(bboxes):
        coord = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_idx = int(bbox[5])
        bbox_color = colors[class_idx]
        bbox_thick = max(int(0.6 * (height + width) / 1000), 1)
        fontScale = 0.75 * bbox_thick

        (x1, y1), (x2, y2) = (coord[0], coord[1]), (coord[2], coord[3])
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)

        if show_label:
            score_str = f'{score:.2f}' if show_confidence else ''
            label = f'{labels[class_idx]}' + score_str

            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=fontScale, thickness=bbox_thick
            )
            cv2.rectangle(image, (x1, y1), (x1+text_w, y1-text_h-baseline), bbox_color, thickness=cv2.FILLED)
            cv2.putText(image, label, (x1, y1-4), \
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)
    return image

def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)


    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])  #w * h
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious

def nms(bboxes, iou_threshold):   
# nms : non maximum suppression 

    """
    bboxes: (xmin, ymin, xmax, ymax, score, class)
    """
    classes_in_img = list(set(bboxes[:, 5]))   # 5 : class_id >> [1,2]
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)   
        cls_bboxes = bboxes[cls_mask]
        # Process 1: Determine whether the number of bounding boxes is greater than 0 
        while len(cls_bboxes) > 0:
            # Process 2: Select the bounding box with the highest score according to socre order A
            max_ind = np.argmax(cls_bboxes[:, 4])   # 4 : confidence score 
            best_bbox = cls_bboxes[max_ind]  # max_ind = 0 
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            # best bbox 선정된 거 외 위/아래 로 재구성 

            # Process 3: Calculate this bounding box A and
            # Remain all iou of the bounding box and remove those bounding boxes whose iou value is higher than the threshold 
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            iou_mask = iou > iou_threshold  #iou_threshold = 0.5
            weight[iou_mask] = 0.0
            
            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

def postprocess_boxes(pred_bbox, original_image, input_size, score_threshold):
    valid_scale=[0, np.inf] # 0과 무한대

    pred_xywh = pred_bbox[:, 0:4]  # (x, y, w, h)
    pred_conf = pred_bbox[:, 4]    # confidence score
    pred_prob = pred_bbox[:, 5:]   # class_probability 

    # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    
    # 2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = original_image.shape[:2]
    resize_ratio = min(input_size / org_w, input_size / org_h)
    # prediction 된 img resize

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # 3. clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)  # 오른쪽 하단의 끝 값
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0   # 범위를 벗어나면 0 

    # 4. discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))
    # rectangle 조건을 만족할 때, scale_mask 

    # 5. discard boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

def detect_image(image, net, output_layers):
    
    pred_bboxes = get_bboxes(image, net, output_layers)
    bboxes = postprocess_boxes(pred_bboxes, image, input_size=target_len, score_threshold=0.25)
    best_bboxes = nms(bboxes, iou_threshold=0.45)
    image_bbox = draw_bbox(image, best_bboxes)

    return image_bbox


# ---------------------------------------------------------


# 실시간 추적 시작
cam = cv2.VideoCapture(0)

pTime = 0
while True:
    success, frame = cam.read()

    if not success: break

    frame_rev = image_preprocess(frame)
    frame_box = detect_image(frame_rev, net, output_layers)
    
    # fps 표시
    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime
    cv2.putText(frame_box, f'FPS: {fps:.2f}', (280, 100), cv2.FONT_HERSHEY_SIMPLEX,
    0.8, (255, 0, 0), 2)

    cv2.imshow('detected', frame_box)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()

