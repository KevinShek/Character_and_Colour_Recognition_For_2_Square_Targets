import cv2
import numpy as np


def detection(frame, config):
    # Initialising variable
    inner_switch = 0
    storing_inner_boxes_data = []

    # if config.Distance_Test:
    #     height, width, _ = frame.shape
    #     if float(config.number) <= 1.0:
    #         frame = frame
    #     elif float(config.number) <= 2.0:
    #         frame = frame[int(height/4):int(3*height/4), int(width/4):int(3*width/4)]
    #     else:
    #         frame = frame[int(height / 3):int(3 * height / 4), int(width / 3):int(3 * width / 4)]

    edged_copy = edge_detection(frame, inner_switch, config)

    # find contours in the threshold image and initialize the
    (contours, _) = cv2.findContours(edged_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # grabs contours

    # try:
    boxes = locating_square(contours, edged_copy, config)
    # except TypeError:
        # if config.capture == "pc" or config.capture == "pi":
            # return _, _, _, _, _, _, _, False
        # else:
            # return _, _, _, edged_copy, _, _, _, False
    
    for i in range(int(len(boxes)/6)):
        possible_target = cv2.rectangle(frame,  # draw rectangle on original testing image
                                (boxes[0+(6*i)], boxes[1+(6*i)]),
                                # upper left corner
                                (boxes[0+(6*i)] + boxes[2+(6*i)],
                                 boxes[1+(6*i)] + boxes[3+(6*i)]),
                                # lower right corner
                                (0, 0, 255),  # green
                                3)
    
        if config.Step_camera:
            rect = cv2.minAreaRect(boxes[5+(6*i)])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
            cv2.imshow("frame", frame)
    
        roi = frame[boxes[1+(6*i)]:boxes[1+(6*i)] + boxes[3+(6*i)], boxes[0+(6*i)]:boxes[0+(6*i)] + boxes[2+(6*i)]]
    
        # Rotating the square to an upright position
        height, width, numchannels = frame.shape
    
        centre_region = (boxes[0+(6*i)] + boxes[2+(6*i)] / 2, boxes[1+(6*i)] + boxes[3+(6*i)] / 2)
    
        # grabs the angle for rotation to make the square level
        angle = cv2.minAreaRect(boxes[4+(6*i)])[-1]  # -1 is the angle the rectangle is at
    
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
        img_rotated = cv2.warpAffine(frame, rotated, (width, height))  # width and height was changed
        img_cropped = cv2.getRectSubPix(img_rotated, (boxes[2+(6*i)], boxes[3+(6*i)]), tuple(centre_region))
    
        # print(f"angle after = {angle}")
    
        if config.square == 2:
            inner_switch = 1
            new_roi = img_cropped[int((boxes[3+(6*i)] / 2) - (boxes[3+(6*i)] / 3)):int((boxes[3+(6*i)] / 2) + (boxes[3+(6*i)] / 3)), int((boxes[2+(6*i)] / 2) - (boxes[2+(6*i)] / 3)):int((boxes[2+(6*i)] / 2) + (boxes[2+(6*i)] / 3))]
            edge = edge_detection(new_roi, inner_switch, config)
            before_edge_search = edge.copy()
            (inner_contours, _) = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # grabs contours
    
            if config.Step_detection:
                cv2.imshow("inner_edge", edge)
                cv2.imshow("testing", frame)
                cv2.waitKey(0)
    
            inner_box = locating_square(inner_contours, edge, config)
            if len(inner_box) == 0:
                if config.capture == "pc" or config.capture == "pi":
                    storing_inner_boxes_data.extend((_, _, _, _, _, _, _, False))
                    continue
                    # return _, _, _, _, _, _, _, False
                else:
                    print("Detection failed to locate the inner square")
                    storing_inner_boxes_data.extend((_, _, _, edged_copy, before_edge_search, edge, possible_target, False))
                    continue
                    # return _, _, _, edged_copy, before_edge_search, edge, possible_target, False
            
            color = new_roi[inner_box[1]:inner_box[1] + inner_box[3], inner_box[0]:inner_box[0] + inner_box[2]]
            print("detected a square target")
    
        elif config.square == 3:
            color = img_cropped[int((boxes[3+(6*i)] / 2) - (boxes[3+(6*i)] / 4)):int((boxes[3+(6*i)] / 2) + (boxes[3+(6*i)] / 4)), int((boxes[2+(6*i)] / 2) - (boxes[2+(6*i)] / 4)):int((boxes[2+(6*i)] / 2) + (boxes[2+(6*i)] / 4))]
            print("detected a square target")
    
        elif config.square == 1:
            color = img_cropped
            print("detected a square target")
    
        if config.Step_detection:
            cv2.imshow("rotated image", img_cropped)
            cv2.imshow("inner square", color)
    
            new = cv2.rectangle(frame,  # draw rectangle on original testing image
                                (boxes[0+(6*i)], boxes[1+(6*i)]),
                                # upper left corner
                                (boxes[0+(6*i)] + boxes[2+(6*i)],
                                 boxes[1+(6*i)] + boxes[3+(6*i)]),
                                # lower right corner
                                (0, 0, 255),  # green
                                3)
            cv2.imshow("frame block", new)
    
        if config.Step_detection:
            cv2.imshow("captured image", roi)
            cv2.waitKey(0)
            
        storing_inner_boxes_data.extend((color, roi, frame, edged_copy, before_edge_search, edge, possible_target, True))
        return storing_inner_boxes_data
    
    if config.capture == "pc" or config.capture == "pi":
        storing_inner_boxes_data.extend((_, _, _, _, _, _, _, False))
        # return _, _, _, _, _, _, _, False
    else:
        storing_inner_boxes_data.extend((_, _, _, edged_copy, _, _, _, False))
        # return _, _, _, edged_copy, _, _, _, False
        
    return storing_inner_boxes_data

def edge_detection(frame, inner_switch, config):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # converts to gray
    if inner_switch == 1:
        blurred_inner = cv2.GaussianBlur(gray, (3, 3), 0)  # blur the gray image for better edge detection
        edged_inner = cv2.Canny(blurred_inner, 5, 5)  # the lower the value the more detailed it would be
        edged = edged_inner
        if config.Step_camera:
            cv2.imshow('edge_inner', edged_inner)
            cv2.imshow("blurred_inner", blurred_inner)

    else:
        blurred_outer = cv2.GaussianBlur(gray, (5, 5), 0)  # blur the gray image for better edge detection
        edged_outer = cv2.Canny(blurred_outer, 20, 20)  # the lower the value the more detailed it would be
        edged = edged_outer
        if config.Step_camera:
            cv2.imshow('edge_outer', edged_outer)
            cv2.imshow("blurred_outer", blurred_outer)
            # cv2.waitKey(0)
    edged_copy = edged
    return edged_copy


def locating_square(contours, edged_copy, config):
    boxes = []
    # outer square
    for c in contours:
        peri = cv2.arcLength(c, True)  # grabs the contours of each points to complete a shape
        # get the approx. points of the actual edges of the corners
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        cv2.drawContours(edged_copy, [approx], -1, (255, 0, 0), 3)
        if config.Step_detection:
            cv2.imshow("contours_approx", edged_copy)

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
    # return boxes[0], boxes[1], boxes[2], boxes[3], boxes[4], boxes[5]
    return boxes

