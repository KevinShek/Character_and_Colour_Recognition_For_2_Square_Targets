import time
import cv2
import numpy as np
import csv
import os
import colour_recognition
import character_recognition
from pathlib import Path
import itertools
from config import Settings
from saving import Saving
from collections import Counter

"""
The following code contains the detection of the square target and saves only the inner square data
"""


def solution(counter, marker, predicted_character, predicted_color, result_dir):
    with open(f'{result_dir}/results.csv', 'a') as csvfile:  # for testing purposes
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        filewriter.writerow([str(marker), str(predicted_character), str(predicted_color)])

    print("detection of marker", marker, "located")
    print(predicted_character + " is located for marker", marker)
    print(predicted_color + " is the colour of ground marker", marker)

    counter = 1
    marker += 1

    return marker, counter


def detection(frame, config):
    # Initialising variable
    inner_switch = 0

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

    try:
        x, y, w, h, approx, cnt = locating_square(contours, edged_copy, config)
    except TypeError:
        return _, _, _, False

    if config.Step_camera:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
        cv2.imshow("frame", frame)

    roi = frame[y:y + h, x:x + w]

    # Rotating the square to an upright position
    height, width, numchannels = frame.shape

    centre_region = (x + w / 2, y + h / 2)

    # grabs the angle for rotation to make the square level
    angle = cv2.minAreaRect(approx)[-1]  # -1 is the angle the rectangle is at

    if 0 == angle:
        angle = angle
    elif -45 > angle > 90:
        angle = -(90 + angle)
    elif -45 > angle:
        angle = 90 + angle
    else:
        angle = angle

    rotated = cv2.getRotationMatrix2D(tuple(centre_region), angle, 1.0)
    img_rotated = cv2.warpAffine(frame, rotated, (width, height))  # width and height was changed
    img_cropped = cv2.getRectSubPix(img_rotated, (w, h), tuple(centre_region))

    if config.square == 2:
        inner_switch = 1
        new_roi = img_cropped[int((h / 2) - (h / 3)):int((h / 2) + (h / 3)), int((w / 2) - (w / 3)):int((w / 2) + (w / 3))]
        edge = edge_detection(new_roi, inner_switch, config)
        (inner_contours, _) = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # grabs contours

        if config.Step_detection:
            cv2.imshow("inner_edge", edge)
            cv2.imshow("testing", frame)
            cv2.waitKey(0)

        try:
            inner_x, inner_y, inner_w, inner_h, approx, _ = locating_square(inner_contours, edged_copy, config)
        except TypeError:
            if config.testing == "detection":
                print("Detection failed to locate the inner square")
            return _, _, _, False
        color = new_roi[inner_y:inner_y + inner_h, inner_x:inner_x + inner_w]
        print("detected a square target")

    elif config.square == 3:
        color = img_cropped[int((h / 2) - (h / 4)):int((h / 2) + (h / 4)), int((w / 2) - (w / 4)):int((w / 2) + (w / 4))]
        print("detected a square target")

    elif config.square == 1:
        color = img_cropped
        print("detected a square target")

    if config.Step_detection:
        cv2.imshow("rotated image", img_cropped)
        cv2.imshow("inner square", color)

        new = cv2.rectangle(frame,  # draw rectangle on original testing image
                            (x, y),
                            # upper left corner
                            (x + w,
                             y + h),
                            # lower right corner
                            (0, 0, 255),  # green
                            3)
        cv2.imshow("frame block", new)

    if config.Step_detection:
        cv2.imshow("captured image", roi)
        cv2.waitKey(0)

    return color, roi, frame, True


def capture_setting():
    # intialising the key information
    counter = 1
    marker = 1
    distance = 0
    predicted_character_list = []
    predicted_color_list = []
    end = time.time()
    start = time.time()
    config = Settings()
    save = Saving(config.name_of_folder, config.exist_ok)

    if config.capture == "pc":
        if config.testing == "video":
            cap = cv2.VideoCapture(config.media)
        else:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)  # 800 default
            cap.set(3, 960)  # 800 default
            cap.set(4, 540)  # 800 default
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            cap.set(cv2.CAP_PROP_FPS, 60)

            time.sleep(2)  # allows the camera to start-up
        print('Camera on')
        while True:
            if counter == 1:
                if config.pause:
                    distance = input("Are you Ready?")
            if counter == 1 or end - start < 10:
                end = time.time()
                ret, frame = cap.read()
                if config.Step_camera:
                    cv2.imshow('frame', frame)
                    k = cv2.waitKey(5) & 0xFF
                    if k == 27:
                        break

                color, roi, frame, success = detection(frame, config)

                if success:
                    counter = counter + 1

                    # time that the target has been last seen
                    start = time.time()

                    predicted_character, contour_image, chosen_image = character_recognition.character(color)
                    predicted_color, processed_image = colour_recognition.colour(color)

                    predicted_character_list.append(predicted_character)
                    predicted_color_list.append(predicted_color)

                    if config.save_results:
                        name_of_results = ["color", "roi", "frame","contour_image","processed_image", "chosen_image"]
                        image_results = [color, roi, frame, contour_image, processed_image, chosen_image]
                        for value, data in enumerate(name_of_results):
                            image_name = f"{marker}_{data}_{counter}.jpg"
                            image = image_results[value]
                            if image is not None:
                                save.save_the_image(image_name, image)

                if counter == 8:
                    print("Starting Recognition Thread")
                    common_character = Counter(predicted_character_list).most_common(1)[0][0]
                    common_color = Counter(predicted_color_list).most_common(1)[0][0]
                    solution(counter, marker, common_character, common_color, save.save_dir)
                    predicted_character_list = []
                    predicted_color_list = []

            else:
                print("Starting Recognition Thread")
                common_character = Counter(predicted_character_list).most_common(1)[0][0]
                common_color = Counter(predicted_color_list).most_common(1)[0][0]
                solution(counter, marker, common_character, common_color, save.save_dir)
                predicted_character_list = []
                predicted_color_list = []

    elif config.capture == "pi":
        from picamera.array import PiRGBArray
        from picamera import PiCamera

        camera = PiCamera()
        camera.resolution = (1280, 720)
        camera.brightness = 50  # 50 is default
        camera.framerate = 90
        camera.awb_mode = 'auto'
        camera.shutter_speed = camera.exposure_speed
        cap = PiRGBArray(camera, size=(1280, 720))

        for image in camera.capture_continuous(cap, format="bgr", use_video_port=True):
            #  to start the progress of capture and don't stop unless the counter increases and has surpass 5 seconds
            if counter == 1 or end - start < 10:
                frame = image.array
                end = time.time()

                color, roi, frame, success = detection(frame, config)
                
                if success:
                    counter = counter + 1

                    # time that the target has been last seen
                    start = time.time()

                    predicted_character, contour_image, chosen_image = character_recognition.character(color)
                    predicted_color, processed_image = colour_recognition.colour(color)

                    predicted_character_list.append(predicted_character)
                    predicted_color_list.append(predicted_color)

                    if config.save_results:
                        name_of_results = ["color", "roi", "frame","contour_image","processed_image", "chosen_image"]
                        image_results = [color, roi, frame, contour_image, processed_image, chosen_image]
                        for value, data in enumerate(name_of_results):
                            image_name = f"{marker}_{data}_{counter}.jpg"
                            image = image_results[value]
                            if image is not None:
                                save.save_the_image(image_name, image)

                if counter == 8:
                    print("Starting Recognition Thread")
                    common_character = Counter(predicted_character_list).most_common(1)[0][0]
                    common_color = Counter(predicted_color_list).most_common(1)[0][0]
                    marker, counter = solution(counter, marker, common_character, common_color, save.save_dir)
                    predicted_character_list = []
                    predicted_color_list = []

            else:
                print("Starting Recognition Thread")
                common_character = Counter(predicted_character_list).most_common(1)[0][0]
                common_color = Counter(predicted_color_list).most_common(1)[0][0]
                marker, counter = solution(counter, marker, common_character, common_color, save.save_dir)
                predicted_character_list = []
                predicted_color_list = []
            cap.truncate(0)
                
    elif config.capture == "image":
        cap = [] # to store the names of the images
        data_dir = Path(config.media)

        # the following code interite over the extension that exist within a folder and place them into a single list
        image_count = list(itertools.chain.from_iterable(data_dir.glob(pattern) for pattern in ('*.jpg', '*.png')))
        # image_count = len(list(data_dir.glob('*.jpg')))
        for name in image_count:
                # head, tail = ntpath.split(name)
                filename = Path(name)  # .stem removes the extension and .name grabs the filename with extension
                cap.append(filename)
                test_image = cv2.imread(str(filename))
                marker = Path(name).stem # grabs the name with the extension

                color, roi, frame, success = detection(test_image, config)

                if success:
                    predicted_character, contour_image, chosen_image = character_recognition.character(color)
                    predicted_color, processed_image = colour_recognition.colour(color)

                    _, _ = solution(counter, marker, predicted_character, predicted_color, save.save_dir)

                    if config.save_results:
                        name_of_results = ["color", "roi", "frame","contour_image","processed_image", "chosen_image", color, roi, frame, contour_image, processed_image, chosen_image]
                        for value in range(5):
                            image_name = f"{marker}_{name_of_results[value]}.jpg"
                            image = name_of_results[value + 6]
                            if image is not None:
                                save.save_the_image(image_name, image)

                    print("Detected and saved a target")
        print(f"there is a total image count of {len(image_count)} and frames appended {len(cap)}")


        # if config.testing == "detection":
        #     if config.Distance_Test:
        #         Characters = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
        #                         'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        #         for j in np.arange(0.5, 6.5, 0.5):
        #             for i in range(0, len(Characters)):
        #                 rel_path = "Test_Images/resolution/{0}/{1}/{2}_{1}_{0}.png".format(config.file_path,
        #                     j, Characters[i])  # where is the image is located from where the config is current
        #                 # located
        #                 config.number = j # distance
        #                 config.dictory = "resolution/{0}/{1}".format(config.file_path, j)
        #                 config.switch = True
        #                 abs_file_path = os.path.join(config.script_dir, rel_path)  # attaching the location
        #                 test_image = cv2.imread(abs_file_path)  # reading in the image
        #                 marker = (os.path.basename(rel_path))
        #                 config.alphanumeric_character = (Characters[i])
        #                 color, roi, frame, success = detection(test_image)
        #                 solution(counter + 1, marker, distance)
        #     else:
        #         color, roi, frame, success = detection(test_image)
        #         predicted_character, contour_image, chosen_image = character_recognition.character(color)
        #         predicted_color, processed_image = colour_recognition.colour(color)
        #         _, _ = solution(counter, marker, predicted_character, predicted_color)

        # elif config.testing == "detection_only":
        #     color, roi, frame, success = detection(test_image)

        # elif config.testing == "character":
        #     predicted_character, _, _ = character_recognition.character(img)
        #     print(str(predicted_character))

        # elif config.testing == "colour":
        #     predicted_color, processed_image = colour_recognition.colour(img)
        #     print(str(predicted_color))


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
        edged_outer = cv2.Canny(blurred_outer, 14, 10)  # the lower the value the more detailed it would be
        edged = edged_outer
        if config.Step_camera:
            cv2.imshow('edge_outer', edged_outer)
            cv2.imshow("blurred_outer", blurred_outer)
            # cv2.waitKey(0)
    edged_copy = edged.copy()
    return edged_copy


def locating_square(contours, edged_copy, config):
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
            keepAspectRatio = 0.6 <= aspectRatio <= 1.4
            if keepDims and keepSolidity and keepAspectRatio:  # checks if the values are true
                return x, y, w, h, approx, c


def main():
    print('Starting detection')
    capture_setting()


if __name__ == "__main__":
    main()
