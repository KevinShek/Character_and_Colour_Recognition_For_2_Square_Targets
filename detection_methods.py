import cv2
import numpy as np
import time
from math import sqrt

class Detection:
    def __init__(self, config):
        self.config = config
        if self.config.detection_method == 1:
            self.loading_ksnn_model()


    def next_frame(self, frame):
        self.frame = frame
        self.storing_inner_boxes_data = []

        if self.config.detection_method == 0:
            self.square_detection()
        elif self.config.detection_method == 1:
            self.ksnn_detection()


    def loading_ksnn_model(self):
        from ksnn.api import KSNN
        self.yolov4 = KSNN('VIM3')
        print(' |---+ KSNN Version: {} +---| '.format(self.yolov4.get_nn_version()))
        print('Start init neural network ...')
        self.yolov4.nn_init(library=self.config.library, model=self.config.model, level=self.config.level)
        print('Done.')


    def organising_pre_data(self, data):
        LISTSIZE = 6 # number of classes + 5
        SPAN = 3

        GRID0 = int(sqrt(len(data[0][2]) / (LISTSIZE * SPAN)))
        GRID1 = int(sqrt(len(data[0][1]) / (LISTSIZE * SPAN)))
        GRID2 = int(sqrt(len(data[0][0]) / (LISTSIZE * SPAN)))

        input0_data = data[0][2]
        input1_data = data[0][1]
        input2_data = data[0][0]


        input0_data = input0_data.reshape(SPAN, LISTSIZE, GRID0, GRID0)
        input1_data = input1_data.reshape(SPAN, LISTSIZE, GRID1, GRID1)
        input2_data = input2_data.reshape(SPAN, LISTSIZE, GRID2, GRID2)

        input_data = list()
        input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
        input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
        input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

        return input_data


    def organising_post_data(self, boxes, classes, scores):
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
            i = torch.ops.torchvision.nms(boxes, scores, self.config.NMS_THRESH)
            print(i)
            if i.shape[0] > self.config.MAX_BOXES:  # limit detections
                i = i[:self.config.MAX_BOXES]

            output[x] = i

        # print(type(output))
        # print(len(output))

        return output


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def process(self, input, mask, anchors):

        anchors = [anchors[i] for i in mask]
        grid_h, grid_w = map(int, input.shape[0:2])

        box_confidence = self.sigmoid(input[..., 4])
        box_confidence = np.expand_dims(box_confidence, axis=-1)

        box_class_probs = self.sigmoid(input[..., 5:])

        box_xy = self.sigmoid(input[..., :2])
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


    def filter_boxes(self, boxes, box_confidences, box_class_probs):

        box_scores = box_confidences * box_class_probs
        box_classes = np.argmax(box_scores, axis=-1)
        box_class_scores = np.max(box_scores, axis=-1)
        pos = np.where(box_class_scores >= self.config.OBJ_THRESH)

        boxes = boxes[pos]
        classes = box_classes[pos]
        scores = box_class_scores[pos]

        return boxes, classes, scores

    
    def nms_boxes(self, boxes, scores):

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
            inds = np.where(ovr <= self.config.NMS_THRESH)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep


    def yolov4_post_process(self, data):
        # use the command line below for vim3pro
        # export LD_PRELOAD=/home/khadas/.local/lib/python3.8/site-packages/torch/lib/libgomp-d22c30c5.so.1
        import torch

        input_data = self.organising_pre_data(data)

        '''YOLOv4-CSP Masks and Anchors'''

        masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55],
                [72, 146], [142, 110], [192, 243], [459, 401]]

        boxes, classes, scores = [], [], []
        for input,mask in zip(input_data, masks):
            b, c, s = self.process(input, mask, anchors)
            b, c, s = self.filter_boxes(b, c, s)
            boxes.append(b)
            classes.append(c)
            scores.append(s)

        boxes = np.concatenate(boxes)
        classes = np.concatenate(classes)
        scores = np.concatenate(scores)

        time_limit = 10.0  # seconds to quit after
        t = time.time()
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]

            keep = self.nms_boxes(b, s)

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
            if xi >= self.config.MAX_BOXES:  # limit detections
                break

            temp_list.append(temp_array)
            
            if (time.time() - t) > time_limit:
                break  # time limit exceeded

        output[nc-1] = torch.Tensor(temp_list)

        return output


    def validation_of_inner_box_for_vim3pro(self, image):
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
        
        self.edge_detection(cropped_roi, inner_switch)
        (inner_contours, _) = cv2.findContours(self.edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # grabs contours
        inner_box = self.locating_square(inner_contours)
        if len(inner_box) == 0:
            inner_switch = 0
        self.edge_detection(roi, inner_switch)
        (inner_contours, _) = cv2.findContours(self.edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # grabs contours
        inner_box = self.locating_square(inner_contours)
        if len(inner_box) == 0:
            return None, False
        else:
            roi = cropped_roi
            
        # rotate the square to an upright position
        height, width, _ = roi.shape
        for i in range(int(len(inner_box)/6)):
            previous_large_area = inner_box[2+(6*i)] * inner_box[3+(6*i)]
            if previous_large_area > current_large_area:
                current_large_area = previous_large_area
                chosen_i = i

        img_cropped = self.rotation_to_upright([inner_box[0+(6*chosen_i)], inner_box[1+(6*chosen_i)], inner_box[2+(6*chosen_i)], inner_box[3+(6*chosen_i)], inner_box[4+(6*chosen_i)], inner_box[5+(6*chosen_i)]])
        
        return img_cropped, True
        

    def draw(self, boxes, scores, classes):
        image_copy = self.frame.copy()

        for box, score, cl in zip(boxes, scores, classes):
            possible_target = self.frame.copy()
            frame = self.frame
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
            x *= self.frame.shape[1]
            y *= self.frame.shape[0]
            w *= self.frame.shape[1]
            h *= self.frame.shape[0]
            # print(f"x,y,w,h = {x}, {y}, {w}, {h}")
            top = max(0, np.floor(x + 0.5).astype(int))
            left = max(0, np.floor(y + 0.5).astype(int))
            right = min(self.frame.shape[1], np.floor(w + 0.5).astype(int))
            bottom = min(self.frame.shape[0], np.floor(h + 0.5).astype(int))
            
            colour = image_copy[left:bottom, top:right]
            
            height, width, _ = colour.shape
            
            if height == 0 or width == 0:
                continue
            
            rotated, valid = self.validation_of_inner_box_for_vim3pro(colour)
            
            # print(f"top,left,right,bottom = {top}, {left}, {right}, {bottom}")
            
            for image_type in [possible_target, frame]:
                cv2.rectangle(image_type, (top, left), (right, bottom), (255, 0, 0), 2)
                cv2.putText(image_type, f'{self.config.CLASSES[cl]} {score:.2f}',
                            (top, left - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)

            self.storing_inner_boxes_data.extend((rotated, frame, None, None, None, None, possible_target, valid))

        return
 

    def edge_detection(self, image, inner_switch):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converts to gray
        if inner_switch == 1:
            blurred_inner = cv2.GaussianBlur(gray, (3, 3), 0)  # blur the gray image for better edge detection
            edged_inner = cv2.Canny(blurred_inner, 5, 5)  # the lower the value the more detailed it would be
            self.edged = edged_inner
            if self.config.Step_camera:
                cv2.imshow('edge_inner', edged_inner)
                cv2.imshow("blurred_inner", blurred_inner)

        else:
            blurred_outer = cv2.GaussianBlur(gray, (5, 5), 0)  # blur the gray image for better edge detection
            edged_outer = cv2.Canny(blurred_outer, 20, 20)  # the lower the value the more detailed it would be
            self.edged = edged_outer
            if self.config.Step_camera:
                cv2.imshow('edge_outer', edged_outer)
                cv2.imshow("blurred_outer", blurred_outer)


    def locating_square(self, contours):
        boxes = []
        # outer square
        for c in contours:
            peri = cv2.arcLength(c, True)  # grabs the contours of each points to complete a shape
            # get the approx. points of the actual edges of the corners
            approx = cv2.approxPolyDP(c, 0.01 * peri, True)
            cv2.drawContours(self.edged, [approx], -1, (255, 0, 0), 3)
            if self.config.Step_detection:
                cv2.imshow("contours_approx", self.edged)

            if 4 <= len(approx) <= 6:
                (x, y, w, h) = cv2.boundingRect(approx)  # gets the (x,y) of the top left of the square and the (w,h)
                aspectRatio = w / float(h)  # gets the aspect ratio of the width to height
                area = cv2.contourArea(c)  # grabs the area of the completed square
                hullArea = cv2.contourArea(cv2.convexHull(c))
                solidity = area / float(hullArea)
                keepDims = w > 10 and h > 10
                keepSolidity = solidity > 0.9  # to check if it's near to be an area of a square
                keepAspectRatio = 0.9 <= aspectRatio <= 1.1
                if keepDims and keepSolidity and keepAspectRatio:  # checks if the values are true
                    boxes.extend((x, y, w, h, approx, c))
                    # cv2.imshow("edged_copy", edged_copy)
        # return boxes[0], boxes[1], boxes[2], boxes[3], boxes[4], boxes[5]
        return boxes
    

    def rotation_to_upright(self, boxes):
        # Rotating the square to an upright position
        height, width, _ = self.frame.shape
    
        centre_region = (boxes[0] + boxes[2] / 2, boxes[1] + boxes[3] / 2)
    
        # grabs the angle for rotation to make the square level
        angle = cv2.minAreaRect(boxes[4])[-1]  # -1 is the angle the rectangle is at
    
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
        img_rotated = cv2.warpAffine(self.frame, rotated, (width, height))  # width and height was changed
        img_cropped = cv2.getRectSubPix(img_rotated, (boxes[2], boxes[3]), tuple(centre_region))

        return img_cropped

    
    def square_validation_for_shape(self, boxes, img_cropped,  roi, before_edge_search_outer_square=None, possible_target=None):
        if self.config.square == 2:
            inner_switch = 1
            new_roi = img_cropped[int((boxes[3] / 2) - (boxes[3] / 3)):int((boxes[3] / 2) + (boxes[3] / 3)), int((boxes[2] / 2) - (boxes[2] / 3)):int((boxes[2] / 2) + (boxes[2] / 3))]
            self.edge_detection(new_roi, inner_switch)
            before_edge_search = self.edged.copy()
            (inner_contours, _) = cv2.findContours(self.edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # grabs contours
    
            if self.config.Step_detection:
                cv2.imshow("inner_edge", self.edged)
                cv2.imshow("testing", self.frame)
                cv2.waitKey(0)
    
            inner_box = self.locating_square(inner_contours)
            if len(inner_box) == 0:
                if self.config.source.isnumeric() or self.config.source.endswith('.txt') or self.config.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')) or self.config.source.endswith('.mp4'):
                    self.storing_inner_boxes_data.extend((_, _, _, _, _, _, _, False))
                    return
                else:
                    print("Detection failed to locate the inner square")
                    self.storing_inner_boxes_data.extend((_, _, _, before_edge_search_outer_square, before_edge_search, self.edged, possible_target, False))
                    return

            
            color = new_roi[inner_box[1]:inner_box[1] + inner_box[3], inner_box[0]:inner_box[0] + inner_box[2]]
            print("detected a square target")
    
        elif self.config.square == 3:
            color = img_cropped[int((boxes[3] / 2) - (boxes[3] / 4)):int((boxes[3] / 2) + (boxes[3] / 4)), int((boxes[2] / 2) - (boxes[2] / 4)):int((boxes[2] / 2) + (boxes[2] / 4))]
            print("detected a square target")
    
        elif self.config.square == 1:
            color = img_cropped
            print("detected a square target")
    
        if self.config.Step_detection:
            cv2.imshow("rotated image", img_cropped)
            cv2.imshow("inner square", color)
    
            new = cv2.rectangle(self.frame,  # draw rectangle on original testing image
                                (boxes[0], boxes[1]),
                                # upper left corner
                                (boxes[0] + boxes[2],
                                 boxes[1] + boxes[3]),
                                # lower right corner
                                (0, 0, 255),  # green
                                3)
            cv2.imshow("frame block", new)
    
        if self.config.Step_detection:
            cv2.imshow("captured image", self.roi)
            cv2.waitKey(0)
            
        self.storing_inner_boxes_data.extend((color, roi, self.frame, before_edge_search_outer_square, before_edge_search, self.edged, possible_target, True))


    def square_detection(self):
        # Initialising variable
        inner_switch = 0
        self.edge_detection(self.frame, inner_switch)
        before_edge_search_outer_square = self.edged.copy()
        # find contours in the threshold image and initialize the
        (contours, _) = cv2.findContours(self.edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # grabs contours
        boxes = self.locating_square(contours)

        for i in range(int(len(boxes)/6)):
            possible_target = cv2.rectangle(self.frame,  # draw rectangle on original testing image
                                    (boxes[0+(6*i)], boxes[1+(6*i)]),
                                    # upper left corner
                                    (boxes[0+(6*i)] + boxes[2+(6*i)],
                                    boxes[1+(6*i)] + boxes[3+(6*i)]),
                                    # lower right corner
                                    (0, 0, 255),  # green
                                    3)
        
            if self.config.Step_camera:
                rect = cv2.minAreaRect(boxes[5+(6*i)])
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(self.frame, [box], 0, (0, 0, 255), 2)
                cv2.imshow("frame", self.frame)
        
            roi = self.frame[boxes[1+(6*i)]:boxes[1+(6*i)] + boxes[3+(6*i)], boxes[0+(6*i)]:boxes[0+(6*i)] + boxes[2+(6*i)]]

            img_cropped = self.rotation_to_upright([boxes[0+(6*i)], boxes[1+(6*i)], boxes[2+(6*i)], boxes[3+(6*i)], boxes[4+(6*i)], boxes[5+(6*i)]])

            self.square_validation_for_shape(boxes, img_cropped, roi, before_edge_search_outer_square, possible_target)
    
        if self.config.source.isnumeric() or self.config.source.endswith('.txt') or self.config.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')) or self.config.source.endswith('.mp4'):
            self.storing_inner_boxes_data.extend((_, _, _, _, _, _, _, False))
            # return _, _, _, _, _, _, _, False
        else:
            self.storing_inner_boxes_data.extend((_, _, _, before_edge_search_outer_square, _, _, _, False))
            # return _, _, _, edged_copy, _, _, _, False
            
        # return self.storing_inner_boxes_data

    
    def ksnn_detection(self):
        from ksnn.types import output_format
        cv_img = list()
        img_resized = cv2.resize(self.frame, (self.config.model_input_size, self.config.model_input_size))
        cv_img.append(img_resized)
        data = np.array([self.yolov4.nn_inference(cv_img, platform='DARKNET', reorder='2 1 0', output_tensor=3, output_format=output_format.OUT_FORMAT_FLOAT32)], dtype="object")
        output = self.yolov4_post_process(data)

        if output is not None and len(output[0]) != 0:
            outputs = output[0].tolist()
            outputs = np.array(outputs)
            boxes = outputs[:, :4]
            scores = outputs[:,4]
            classes = outputs[:,5]
            classes_int = classes.astype(int)
            
            self.draw(boxes, scores, classes_int)
        else:
            self.storing_inner_boxes_data.extend((None, self.frame, None, None, None, None, None, False))
            
        # return self.storing_inner_boxes_data
