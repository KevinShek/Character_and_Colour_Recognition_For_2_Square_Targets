import time
from math import *
import cv2
import numpy as np
import imutils.video
from collections import Counter
import webcolors
import operator
import csv
import os
import threading
import GPS
import colour_recognition as colour_recognition
import character_recognition as character_recognition
from config import Settings
if Settings.capture == "pi":
    from picamera.array import PiRGBArray
    from picamera import PiCamera

"""
The following code contains the detection of the square target and saves only the inner square data
"""


def solution(counter, marker, distance):
    if not Settings.Distance_Test:
        print("detection of marker", marker, "located")
        print(character_recognition.character(counter, marker, distance) +
              " is located for marker", marker)
        print(colour_recognition.colour(counter, marker, distance) + " is the colour of ground marker",
              marker)
    else:
        print(marker)

    if Settings.Save_Data or Settings.Static_Test:
        with open('results.csv', 'a') as csvfile:  # for testing purposes
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            filewriter.writerow(
                [str(marker), str(character_recognition.character(counter, marker, distance)),
                 str(colour_recognition.colour(counter, marker, distance))])

    if Settings.Distance_Test:
        if Settings.switch:
            name = "{0}_{1}_{2}".format(Settings.file_path, Settings.number, Settings.character)
            if character_recognition.character(counter, marker, distance) == Settings.alphanumeric_character:
                comparison = 1
            else:
                comparison = 0
            with open('{0}.csv'.format(name), 'a') as csvfile:  # for testing purposes
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                filewriter.writerow(
                    [str(marker), 1, str(character_recognition.character(counter, marker, distance)), comparison,
                     str(colour_recognition.colour(counter, marker, distance))])
        else:
            name = "{0}_{1}_{2}".format(Settings.file_path, Settings.number, Settings.character)
            with open('{0}.csv'.format(name), 'a') as csvfile:  # for testing purposes
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                filewriter.writerow([str(marker), 0, 0, 0])

    if Settings.GPS and Settings.air == 1:
        # Get middle position within lists outputted by detection function
        middle = int(len(positions) / 2)

        print(
            GPS.GPS(centres[middle], headings[middle], positions[middle][0], positions[middle][1], positions[middle][2],
                    height_of_target[middle]) + " latitude and longitude of", marker)

    elif Settings.GPS and Settings.air != 1:
        # Get middle position within lists outputted by detection function
        middle = int(len(positions) / 2)

        print(
            GPS.GPS(centres[middle], headings[middle], positions[middle][0], positions[middle][1], positions[middle][2],
                    height_of_target[middle]) + " latitude, longitude and altitidue of", marker)

    if not Settings.Distance_Test:
        marker += 1  # Add one to its original value
    counter = 1

    return marker, counter


def detection(frame, counter, marker, distance):
    # Initialising variable
    positions = []
    headings = []
    centres = []
    height_of_target = []
    directory = None

    # Gathering data from Pixhawk
    if Settings.GPS:
        position = vehicle.location.global_relative_frame
        heading = vehicle.heading
    # end if

    inner_switch = 0

    if Settings.Distance_Test:
        height, width, _ = frame.shape
        if float(Settings.number) <= 1.0:
            frame = frame
        elif float(Settings.number) <= 2.0:
            frame = frame[int(height/4):int(3*height/4), int(width/4):int(3*width/4)]
        else:
            frame = frame[int(height / 3):int(3 * height / 4), int(width / 3):int(3 * width / 4)]

    edged_copy = edge_detection(frame, inner_switch)

    # find contours in the threshold image and initialize the
    (contours, _) = cv2.findContours(edged_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # grabs contours

    try:
        x, y, w, h, approx = locating_square(contours, edged_copy)
    except TypeError:
        Settings.switch = False
        return False

    roi = frame[y:y + h, x:x + w]

    # Rotating the square to an upright position
    height, width, numchannels = frame.shape

    centre_region = (x + w / 2, y + h / 2)
    if Settings.GPS:
        centre_target = (y + h / 2, x + w / 2)

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

    if Settings.square == 2:
        inner_switch = 1
        new_roi = img_cropped[int((h / 2) - (h / 3)):int((h / 2) + (h / 3)), int((w / 2) - (w / 3)):int((w / 2) + (w / 3))]
        edge = edge_detection(new_roi, inner_switch)
        (inner_contours, _) = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # grabs contours

        if Settings.Step_detection:
            cv2.imshow("inner_edge", edge)
            cv2.imshow("testing", frame)
            cv2.waitKey(0)

        try:
            inner_x, inner_y, inner_w, inner_h, approx = locating_square(inner_contours, edged_copy)
        except TypeError:
            if Settings.testing == "detection":
                print("Detection failed to locate the inner square")
                Settings.switch = False
            return False
        color = new_roi[inner_y:inner_y + inner_h, inner_x:inner_x + inner_w]

    elif Settings.square == 3:
        color = img_cropped[int((h / 2) - (h / 4)):int((h / 2) + (h / 4)), int((w / 2) - (w / 4)):int((w / 2) + (w / 4))]

    elif Settings.square == 1:
        color = img_cropped

    if Settings.Step_detection:
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

    # appends the data of the image to the list
    if Settings.GPS:
        positions.append([position.lat, position.lon, position.alt])
        headings.append(heading)
        centres.append(centre_target)
        height_of_target.append(h)

    cv2.imwrite("colour%d.png" % counter, color)

    if Settings.Save_Data:
        directory = "results"
    elif Settings.Static_Test:
        directory = "{0}".format(distance)
    elif Settings.Rover_Marker:
        directory = "marker={0}".format(marker)
    elif Settings.Distance_Test:
        directory = "{0}".format(Settings.dictory)
        counter = Settings.counter

    write_to_file(directory, marker, counter, "results", color)
    write_to_file(directory, marker, counter, "captured", roi)
    write_to_file(directory, marker, counter, "frame", frame)

    print("Detected and saved a target")

    if Settings.Step_detection:
        cv2.imshow("captured image", roi)
        cv2.waitKey(0)

    return True


def capture_setting():
    counter = 1
    marker = 1
    distance = 0
    end = time.time()
    start = time.time()

    if Settings.capture == "pc":
        if Settings.testing == "video":
            cap = cv2.VideoCapture(Settings.video)
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
                if Settings.Static_Test:
                    distance = input("Distance it was taken")
            if counter == 1 or end - start < 5:
                end = time.time()
                ret, frame = cap.read()
                if Settings.Step_camera:
                    cv2.imshow('frame', frame)
                    k = cv2.waitKey(5) & 0xFF
                    if k == 27:
                        break

                if detection(frame, counter, marker, distance):
                    counter = counter + 1

                    # time that the target has been last seen
                    start = time.time()

                if counter == 8:
                    marker, counter = solution(counter, marker, distance)

            else:
                marker, counter = solution(counter, marker, distance)

    elif Settings.capture == "pi":
        camera = PiCamera()
        camera.resolution = (1280, 720)
        camera.brightness = 50  # 50 is default
        camera.framerate = 90
        camera.awb_mode = 'auto'
        camera.shutter_speed = camera.exposure_speed
        cap = PiRGBArray(camera, size=(1280, 720))

        for image in camera.capture_continuous(cap, format="bgr", use_video_port=True):
            if counter == 1 or end - start < 5:
                frame = image.array
                end = time.time()

                if detection(frame, counter, marker, distance):
                    counter = counter + 1

                    # time that the target has been last seen
                    start = time.time()

                if counter == 8:
                    marker, counter = solution(counter, marker, distance)

            else:
                marker, counter = solution(counter, marker, distance)

        cap.truncate(0)
    elif Settings.capture == "image":
        if Settings.testing == "detection":
            if Settings.Distance_Test:
                Characters = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                              'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
                for j in np.arange(0.5, 6.5, 0.5):
                    for i in range(0, len(Characters)):
                        rel_path = "Test_Images/resolution/{0}/{1}/{2}_{1}_{0}.png".format(Settings.file_path,
                            j, Characters[i])  # where is the image is located from where the config is current
                        # located
                        Settings.number = j
                        Settings.dictory = "resolution/{0}/{1}".format(Settings.file_path, j)
                        Settings.switch = True
                        abs_file_path = os.path.join(Settings.script_dir, rel_path)  # attaching the location
                        test_image = cv2.imread(abs_file_path)  # reading in the image
                        marker = (os.path.basename(rel_path))
                        Settings.alphanumeric_character = (Characters[i])
                        detection(test_image, counter, marker, distance)
                        solution(counter + 1, marker, distance)
            else:
                detection(Settings.test_image, counter, marker, distance)
                solution(counter + 1, marker, distance)
        elif Settings.testing == "character":
            print(character_recognition.character(counter + 1, marker, distance))

        elif Settings.testing == "colour":
            print(colour_recognition.colour(counter + 1, marker, distance))


def write_to_file(directory, marker, k, name, image):
    """
  For writing an image file to the specified directory with standardised file name format
  - directory - folder to save file
  - marker - the target number
  - k - the number of image being viewed
  - image - saving the image of interest
  """

    if directory is not None:
        # Make sure the directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Form file name
        filename = f"{directory}/{marker}_{k}{name}.png"

        # Form full path
        filepath = os.path.join(Settings.script_dir, filename)

        # Write file
        cv2.imwrite(filepath, image)


def edge_detection(frame, inner_switch):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # converts to gray
    if inner_switch == 1:
        blurred_inner = cv2.GaussianBlur(frame, (5, 5), 0)  # blur the gray image for better edge detection
        edged_inner = cv2.Canny(blurred_inner, 14, 10)  # the lower the value the more detailed it would be
        edged = edged_inner
        if Settings.Step_camera:
            cv2.imshow('edge_inner', edged_inner)
            cv2.imshow("blurred_inner", blurred_inner)
            cv2.waitKey(0)
    else:
        blurred_outer = cv2.GaussianBlur(gray, (5, 5), 0)  # blur the gray image for better edge detection
        edged_outer = cv2.Canny(blurred_outer, 14, 10)  # the lower the value the more detailed it would be
        edged = edged_outer
        if Settings.Step_camera:
            cv2.imshow('edge_outer', edged_outer)
            cv2.imshow("blurred_outer", blurred_outer)
            cv2.waitKey(0)
    edged_copy = edged.copy()
    return edged_copy


def locating_square(contours, edged_copy):
    # outer square
    for c in contours:
        peri = cv2.arcLength(c, True)  # grabs the contours of each points to complete a shape
        # get the approx. points of the actual edges of the corners
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        cv2.drawContours(edged_copy, [approx], -1, (255, 0, 0), 3)
        if Settings.Step_detection:
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
                return x, y, w, h, approx


def main():
    print('Starting detection')
    if Settings.GPS:
        print('Connecting to drone...')

        # Connect to vehicle and print some info
        vehicle = connect('192.168.0.156:14550', wait_ready=True, baud=921600)

        print('Connected to drone')
        print('Autopilot Firmware version: %s' % vehicle.version)
        print('Global Location: %s' % vehicle.location.global_relative_frame)

    capture_setting()


if __name__ == "__main__":
    main()
