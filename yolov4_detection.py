import numpy as np
import os
import urllib.request
from matplotlib import gridspec
from matplotlib import pyplot as plt
from PIL import Image
import argparse
import sys
import math
from ksnn.api import KSNN
from ksnn.types import output_format
import cv2 as cv
import time
import torch
import torchvision
from math import sqrt
import math
import cv2

# GRID0 = 16 # 512
# GRID1 = 32 # 512
# GRID2 = 64 # 512

# GRID0 = 20 # 640
# GRID1 = 40 # 640
# GRID2 = 80 # 640

NUM_CLS = 1
MAX_BOXES = 50

CLASSES = "box"


def organising_pre_data(data):

    LISTSIZE = 6 # number of classes + 5
    SPAN = 3

    GRID0 = int(sqrt(len(data[0][2]) / (LISTSIZE * SPAN)))
    GRID1 = int(sqrt(len(data[0][1]) / (LISTSIZE * SPAN)))
    GRID2 = int(sqrt(len(data[0][0]) / (LISTSIZE * SPAN)))

    input0_data = data[0][2]
    input1_data = data[0][1]
    input2_data = data[0][0]
    # print("array0= ", len(data[0][0]))
    # print("array1= ", len(data[0][1]))
    # print("array2= ", len(data[0][2]))

    input0_data = input0_data.reshape(SPAN, LISTSIZE, GRID0, GRID0)
    input1_data = input1_data.reshape(SPAN, LISTSIZE, GRID1, GRID1)
    input2_data = input2_data.reshape(SPAN, LISTSIZE, GRID2, GRID2)

    input_data = list()
    input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
    input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
    input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

    return input_data


def organising_post_data(boxes, classes, scores, NMS_THRESH):
    nc = 0
    for x in classes:
        if x == 0:
            previous_x = x
            nc = 1
        elif x != previous_x:
            nc += 1
        else: 
            continue

    output = [torch.zeros(0, 6)] * nc

    for x in range(nc):
        c = classes[x] * 4096  # classes
        # print("boxes=",boxes)
        boxes, scores = boxes + c, scores  # boxes (offset by class), scores
        # print("boxes=",boxes)
        i = torch.ops.torchvision.nms(boxes, scores, NMS_THRESH)
        print(i)
        if i.shape[0] > MAX_BOXES:  # limit detections
            i = i[:MAX_BOXES]

        output[x] = i

    # print(type(output))
    # print(len(output))

    return output


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def process(input, mask, anchors):

    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = sigmoid(input[..., 5:])

    box_xy = sigmoid(input[..., :2])
    box_wh = np.exp(input[..., 2:4])
    box_wh = box_wh * anchors

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)

    box_xy += grid
    box_xy /= (grid_w, grid_h)
    box_wh /= (416, 416)
    box_xy -= (box_wh / 2.)
    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs

def filter_boxes(boxes, box_confidences, box_class_probs, OBJ_THRESH):

    box_scores = box_confidences * box_class_probs
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    pos = np.where(box_class_scores >= OBJ_THRESH)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores, NMS_THRESH):

    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def yolov4_post_process(data, OBJ_THRESH=0.1, NMS_THRESH=0.6):

    input_data = organising_pre_data(data)

    '''YOLOv4-CSP Masks and Anchors'''

    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55],
            [72, 146], [142, 110], [192, 243], [459, 401]]

    boxes, classes, scores = [], [], []
    for input,mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s, OBJ_THRESH)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    # boxes = torch.from_numpy(boxes)
    # classes = torch.from_numpy(classes.astype(np.float64))
    # scores = torch.from_numpy(scores.astype(np.float64))

    # output = organising_post_data(boxes, classes, scores, NMS_THRESH)

    time_limit = 10.0  # seconds to quit after
    t = time.time()
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s, NMS_THRESH)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        # return None, None, None
        output = [torch.zeros(0, 6)] * 1 
        return output
    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    # print("b=",boxes.dtype, " shape=", boxes.shape)
    # print("c=",classes.dtype, " shape=", classes.shape)
    # print("s=",scores.dtype, " shape=", scores.shape)

    boxes = torch.from_numpy(boxes)
    classes = torch.from_numpy(classes.astype(np.float32))
    scores = torch.from_numpy(scores.astype(np.float32))

    nc = 0
    for x in classes:
        if x == 0:
            previous_x = x
            nc = 1
        elif x != previous_x:
            nc += 1
        else: 
            continue

    output = [torch.zeros(0, 6)] * nc 
    temp_list = []
    for xi in range(len(boxes)):
        temp_array = [boxes[xi][0], boxes[xi][1], boxes[xi][0] + boxes[xi][2], boxes[xi][1] + boxes[xi][3], scores[xi], classes[xi]] # x1, y1, x2, y2, conf, cls
        # temp_array = [boxes[x], scores[x], classes[x]]
        if xi >= MAX_BOXES:  # limit detections
            break

        temp_list.append(temp_array)
        
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    output[nc-1] = torch.Tensor(temp_list)

    # output = torch.Tensor(output)    # 300 prediction, 6 values == (300,6) or (N, 6) where N is the number of prediction
    # print(len(output))
    # print(output.size())
    # print(output[:, 0])
    # img_shape = (640,640)
    # print(output[:, 0].clamp_(0, img_shape[1]))  # x1
    # print(output[:, 1].clamp_(0, img_shape[0]))  # y1
    # print(output[:, 2].clamp_(0, img_shape[1]))  # x2
    # print(output[:, 3].clamp_(0, img_shape[0]))  # y2

    # print("b=",boxes.dtype, " shape=", boxes.size())
    # print("c=",classes.dtype, " shape=", classes.size())
    # print("s=",scores.dtype, " shape=", scores.size())

    return output


def validation_of_inner_box_for_vim3pro(image, frame, config):
    from shape_detection import locating_square, edge_detection
    inner_switch = 1
    current_large_area = 0
    height, width, _ = image.shape
    print(image.shape)
    if height < 100 or width < 100:
        roi = cv2.resize(image, (500, 500))
    else:
        roi = image
    
    # check if there is a inner square already
    height, width, _ = roi.shape
    # print(f"non-cropped roi = {roi.shape}")
    # cv2.imshow("roi", roi)
    # cropped_roi = roi[int(height-((height*3/2))):int(height+(height*3/2)), int(width-(width*3/2)):int(width+(width*3/2))]
    cropped_roi = roi[int(height/5):int(height*4/5), int(width/5):int(width*4/5)]
    # cv2.imshow("current_roi", cropped_roi)
    # print(f"cropped roi = {cropped_roi.shape}")
    # cv2.waitKey(0)
    
    edge = edge_detection(cropped_roi, inner_switch, config)
    (inner_contours, _) = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # grabs contours
    inner_box = locating_square(inner_contours, edge, config)
    if len(inner_box) == 0:
      inner_switch = 0
      edge = edge_detection(roi, inner_switch, config)
      (inner_contours, _) = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # grabs contours
      inner_box = locating_square(inner_contours, edge, config)
      if len(inner_box) == 0:
          return None, False
    else:
      roi = cropped_roi
        
    # rotate the square to an upright position
    height, width, numchannels = roi.shape
    for i in range(int(len(inner_box)/6)):
        previous_large_area = inner_box[2+(6*i)] * inner_box[3+(6*i)]
        if previous_large_area > current_large_area:
            current_large_area = previous_large_area
            chosen_i = i
    centre_region = (inner_box[0+(6*chosen_i)] + inner_box[2+(6*chosen_i)] / 2, inner_box[1+(6*chosen_i)] + inner_box[3+(6*chosen_i)] / 2)

    # grabs the angle for rotation to make the square level
    angle = abs(cv2.minAreaRect(inner_box[4+(6*chosen_i)])[-1])  # -1 is the angle the rectangle is at
    # print(f"{centre_region} {angle}")
    
    # print(f"angle before = {angle}")
    
    if angle == 0.0:
        angle = angle
    elif angle == 180 or angle == -180 or angle == 90 or angle == -90:
        angle = 0.0
    elif angle > 45:
        angle = 90 - angle
    else:
        angle = angle
    
    rotated = cv2.getRotationMatrix2D(tuple(centre_region), angle, 1.0)
    img_rotated = cv2.warpAffine(roi, rotated, (width, height))  # width and height was changed
    img_cropped = cv2.getRectSubPix(img_rotated, (inner_box[2+(6*chosen_i)], inner_box[3+(6*chosen_i)]), tuple(centre_region))
    
    # cv2.imshow("img_rotated", img_rotated)
    # cv2.imshow("img_cropped", img_cropped)
      
      
    return img_cropped, True
    

def draw(image, boxes, scores, classes, config):
    storing_boxes_data = []
    image_copy = image.copy()

    for box, score, cl in zip(boxes, scores, classes):
        possible_target = image.copy()
        frame = image
        if score < 0.3:
          continue
        x, y, w, h = box
        if w > 1:
          w = 1
        if h > 1:
          h = 1
        x, y = abs(x), abs(y)
        # print('class: {}, score: {}'.format(CLASSES[cl], score))
        # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(x, y, w, h))
        x *= image.shape[1]
        y *= image.shape[0]
        w *= image.shape[1]
        h *= image.shape[0]
        # print(f"x,y,w,h = {x}, {y}, {w}, {h}")
        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(h + 0.5).astype(int))
        
        colour = image_copy[left:bottom, top:right]
        
        height, width, _ = colour.shape
        
        if height == 0 or width == 0:
            continue
        
        rotated, valid = validation_of_inner_box_for_vim3pro(colour, image, config)
        
        # print(f"top,left,right,bottom = {top}, {left}, {right}, {bottom}")
        
        for image_type in [possible_target, frame]:
            cv.rectangle(image_type, (top, left), (right, bottom), (255, 0, 0), 2)
            cv.putText(image_type, f'{CLASSES[cl]} {score:.2f}',
                        (top, left - 6),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

        # cv2.imshow("frame", frame)
        # cv2.imshow("colour", colour)
        # if valid:
            # cv2.imshow("rotated", rotated)
        # cv2.waitKey(0)
        
        # print(valid)

        storing_boxes_data.extend((rotated, frame, None, None, None, None, possible_target, valid))

    return storing_boxes_data


def loading_model(config):
    from ksnn.api import KSNN
    yolov4 = KSNN('VIM3')
    print(' |---+ KSNN Version: {} +---| '.format(yolov4.get_nn_version()))
    print('Start init neural network ...')
    yolov4.nn_init(library=config.library, model=config.model, level=config.level)
    config.loaded_model = yolov4
    print('Done.')
    

def detection(image, config):
    cv_img = list()
    img_resized = cv.resize(image, config.model_input_size)
    cv_img.append(img_resized)
    yolov4 = config.loaded_model

    data = np.array([yolov4.nn_inference(cv_img, platform='DARKNET', reorder='2 1 0', output_tensor=3, output_format=output_format.OUT_FORMAT_FLOAT32)], dtype="object")
    output = yolov4_post_process(data, config.OBJ_THRESH, config.NMS_THRESH)
    # print(len(output[0]))
    
    if output is not None and len(output[0]) != 0:
        # print(f"before={output}")
        outputs = output[0].tolist()
        outputs = np.array(outputs)
        # print(len(outputs))
        # print(f"after={outputs}")
        boxes = outputs[:, :4]
        # print(boxes)
        scores = outputs[:,4]
        classes = outputs[:,5]
        classes_int = classes.astype(int)
        
        storing_boxes_data = draw(image, boxes, scores, classes_int, config)
    else:
        storing_boxes_data = []
        storing_boxes_data.extend((None, image, None, None, None, None, None, False))
        
    return storing_boxes_data
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--library", help="Path to C static library file")
    parser.add_argument("--model", help="Path to nbg file")
    parser.add_argument("--picture", help="Path to input picture")
    parser.add_argument("--level", help="Information printer level: 0/1/2")

    args = parser.parse_args()
    if args.model :
        if os.path.exists(args.model) == False:
            sys.exit('Model \'{}\' not exist'.format(args.model))
        model = args.model
    else :
        sys.exit("NBG file not found !!! Please use format: --model")
    if args.picture :
        if os.path.exists(args.picture) == False:
            sys.exit('Input picture \'{}\' not exist'.format(args.picture))
        picture = args.picture
    else :
        sys.exit("Input picture not found !!! Please use format: --picture")
    if args.library :
        if os.path.exists(args.library) == False:
            sys.exit('C static library \'{}\' not exist'.format(args.library))
        library = args.library
    else :
        sys.exit("C static library not found !!! Please use format: --library")
    if args.level == '1' or args.level == '2' :
        level = int(args.level)
    else :
        level = 0

    yolov4 = KSNN('VIM3')
    print(' |---+ KSNN Version: {} +---| '.format(yolov4.get_nn_version()))

    print('Start init neural network ...')
    yolov4.nn_init(library=library, model=model, level=level)
    print('Done.')

    print('Get input data ...')
    cv_img =  list()
    img = cv.imread(picture, cv.IMREAD_COLOR)
    img_resized = cv.resize(img, (640, 640))
    cv_img.append(img)
    print('Done.')
    print(img)

    print('Start inference ...')
    start = time.time()

    '''
        default input_tensor is 1
    '''
    data = np.array([yolov4.nn_inference(img_resized, platform='DARKNET', reorder='2 1 0', output_tensor=3, output_format=output_format.OUT_FORMAT_FLOAT32)])
    end = time.time()
    print('Done. inference time: ', end - start)

    output = yolov4_post_process(data)

    if output is not None:
        outputs = output[0].tolist()
        outputs = np.array(outputs)
        # print(len(outputs))
        # print(outputs[0])
        # print(output[0][:4])
        # print(output[0][4])
        # print(output[0][5])
        boxes = outputs[:, :4]
        scores = outputs[:,4]
        classes = outputs[:,5]
        classes_int = classes.astype(int)
        
        storing_boxes_data = draw(img, boxes, scores, classes_int)
        
        # print(len(storing_boxes_data))
        
    from saving import Saving
    save = Saving("yolov4_results", True)
    
    for i in range(int(len(storing_boxes_data)/8)):  
        if storing_boxes_data[7+(8*i)]:
            name_of_results = ["color", "roi", "frame","contour_image","processed_image", "chosen_image", "outer_edge", "inner_edge", "possible_target", "before_inner_edge_search"]
            image_results = [storing_boxes_data[0+(8*i)], storing_boxes_data[1+(8*i)], storing_boxes_data[2+(8*i)], None, None, None, storing_boxes_data[3+(8*i)], storing_boxes_data[5+(8*i)], storing_boxes_data[6+(8*i)], storing_boxes_data[4+(8*i)]]
            for value, data in enumerate(name_of_results):
                image_name = f"{data}_{i}.jpg"
                image = image_results[value]
                if image is not None:
                    save.save_the_image(image_name, image)
          
        
        
        # for i in range(len(output)):
            # draw(img, outputs[i][:4], outputs[i][4], outputs[i][5])

    # for ind in range(len(boxes)):
    #     print(f"boxes={boxes[ind]}, classes={classes[ind]}, scores={scores[ind]}\n")

    # cv.imwrite("results/results.jpg", img)
    # cv.waitKey(0)
