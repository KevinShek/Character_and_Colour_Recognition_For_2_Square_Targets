import cv2
import numpy as np

class Detection:
    def __init__(self, config, frame):
        self.config = config
        self.frame = frame
        self.storing_inner_boxes_data = []
        
        if self.config.detection_method == "shape":
            shape_detection


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
            cv2.drawContours(self.edged_copy, [approx], -1, (255, 0, 0), 3)
            if self.config.Step_detection:
                cv2.imshow("contours_approx", self.edged_copy)

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
                if self.config.capture == "pc" or self.config.capture == "pi":
                    self.storing_inner_boxes_data.extend((_, _, _, _, _, _, _, False))
                else:
                    print("Detection failed to locate the inner square")
                    self.storing_inner_boxes_data.extend((_, _, _, before_edge_search_outer_square, before_edge_search, self.edged, possible_target, False))

            
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
        before_edge_search_outer_square = self.edged_copy.copy()
        # find contours in the threshold image and initialize the
        (contours, _) = cv2.findContours(self.edged_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # grabs contours
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

            self.square_validation_for_shape(self, boxes, img_cropped, roi, before_edge_search_outer_square, possible_target)
    
        if self.config.capture == "pc" or self.config.capture == "pi":
            self.storing_inner_boxes_data.extend((_, _, _, _, _, _, _, False))
            # return _, _, _, _, _, _, _, False
        else:
            self.storing_inner_boxes_data.extend((_, _, _, before_edge_search_outer_square, _, _, _, False))
            # return _, _, _, edged_copy, _, _, _, False
            
        return self.storing_inner_boxes_data

    

    

