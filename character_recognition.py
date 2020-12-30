import math
import cv2
import numpy as np
from collections import Counter
import operator
import os
import pytesseract
from config import Settings

"""
Recognise a white character from an saved image through the method of KNN or tesseract
"""

# to initialise tesseract
if Settings.device_for_tesseract == "pc":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    TESSDATA_PREFIX = r"C:\Program Files\Tesseract-OCR"
elif Settings.device_for_tesseract == "pi":
    TESSDATA_PREFIX = r"\usr\share\tesseract-ocr"

# Load Character Contour Area
MIN_CONTOUR_AREA = 100

class ContourWithData:
    # member variables ############################################################################
    npaContour = None  # contour
    boundingRect = None  # bounding rect for contour
    intRectX = 0  # bounding rect top left corner x location
    intRectY = 0  # bounding rect top left corner y location
    intRectWidth = 0  # bounding rect width
    intRectHeight = 0  # bounding rect height
    fltArea = 0.0  # area of contour
    intCentreX = 0
    intCentreY = 0

    # calculate bounding rect information
    def bounding_rect_info(self):
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intCentreX = intX / 2
        self.intCentreY = intY / 2
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight
        self.fltDiagonalSize = math.sqrt((self.intRectWidth ** 2) + (self.intRectHeight ** 2))

    # this is oversimplified, for a production grade program, checks if the contour would fall in range of the
    # contour area and any smaller or bigger is treated as noises and ignored
    def contour_valid(self, height, width):
        MAX_CONTOUR_AREA = height * width * 0.9
        return MIN_CONTOUR_AREA < self.fltArea < MAX_CONTOUR_AREA


def character(counter, marker, distance):
    print('Starting recognition thread')

    # initialising variables
    guesses = [0] * 35  # create a list of 35 characters
    text = [] * 7  # create a list of 7 images
    character_prediction = None
    directory = None

    for i in range(1, counter):
        if Settings.character_test:
            img = Settings.test_image
        else:
            img = cv2.imread("colour%d.png" % i)

        # if Step_letter:
        #   plt.hist(gauss.ravel(), bins=256, range=[0.0, 256.0], fc='k', ec='k')  # calculating histogram
        #   plt.show()

        # make a copy of the thresh image, this in necessary b/c findContours modifies the image
        img_thresh_copy = pre_processing(img).copy()

        if Settings.Save_Data:
            directory = "threshold"
        elif Settings.Static_Test:
            directory = "{0}".format(distance)
        elif Settings.Rover_Marker:
            directory = "marker={0}".format(marker)
        elif Settings.Distance_Test:
            directory = "{0}".format(Settings.dictory)

        write_to_file(directory, marker, i, "contour", img_thresh_copy)

        if Settings.character == "knn":
            npaROIResized = area_of_region_of_interest(img_thresh_copy, distance, marker, i)

            knn = cv2.ml.KNearest_create()  # initialise the knn
            # joins the train data with the train_labels
            knn.train(Settings.npaFlattenedImages, cv2.ml.ROW_SAMPLE, Settings.Classifications)

            # looks for the 3 nearest neighbours comparing to the flatten images (k = neighbours)
            retval, npaResults, neigh_resp, dists = knn.findNearest(npaROIResized, k=Settings.knn_value)

            # current guess
            currGuess = int(npaResults[0][0])
            if Settings.Step_letter:
                print(currGuess)
            # Transform guess in ASCII format into range 0-35
            if 49 <= currGuess <= 57:
                guesses[currGuess - 49] += 1
            elif 65 <= currGuess <= 90:
                guesses[currGuess - 56] += 1

        elif Settings.character == "tesseract":
            # configuration setting to convert image to string.
            configuration = "-l eng --oem 3 --psm 10"

            # This will recognize the text from the image of bounding box
            charactername = pytesseract.image_to_string(img_thresh_copy, config=configuration)

            text.append(charactername[:-2])

        else:
            print("please chose between one of the preprocess method of 'knn' or 'tesseract'.")

    if Settings.character == "knn":
        # find mode of character guess
        # Initialise mode and prev variables for first loop through
        if Settings.Step_letter:
            print(guesses)
        mode = 0
        prev = guesses[0]
        for j in range(35):
            new = guesses[j]
            if new > prev:
                prev = guesses[j]
                mode = j
        # Transform back into ASCII
        if 0 <= mode <= 8:
            mode = mode + 49
        elif 9 <= mode <= 34:
            mode = mode + 56

        character_prediction = chr(mode)

    elif Settings.character == "tesseract":
        mode = Counter(text)

        if Settings.Step_letter:
            print(mode)

        if mode == Counter():
            character_prediction = "None"
        else:
            character_prediction = mode.most_common(1)[0][0]

    return character_prediction


def remove_inner_overlapping_chars(list_of_matching_chars):
    # if we have two chars overlapping or to close to each other to possibly be separate chars, remove the inner (
    # smaller) char, this is to prevent including the same char twice if two contours are found for the same char,
    # for example for the letter 'O' both the inner ring and the outer ring may be found as contours, but we should
    # only include the char once
    listOfMatchingCharsWithInnerCharRemoved = list(list_of_matching_chars)  # this will be the return value

    for currentChar in list_of_matching_chars:
        for otherChar in list_of_matching_chars:
            if currentChar != otherChar:  # if current char and other char are not the same char . . .
                # if current char and other char have center points at almost the same location . . .
                if distance_between_chars(currentChar, otherChar) < (currentChar.fltDiagonalSize * 0.3):
                    # if we get in here we have found overlapping chars next we identify which char is smaller,
                    # then if that char was not already removed on a previous pass, remove it
                    if currentChar.fltArea < otherChar.fltArea:  # if current char is smaller than other char
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved:  # if current char was not already
                            # removed on a previous pass . . .
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)  # then remove current char
                        # end if
                    else:  # else if other char is smaller than current char
                        if otherChar in listOfMatchingCharsWithInnerCharRemoved:  # if other char was not already
                            # removed on a previous pass . . .
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)  # then remove other char
                        # end if
                    # end if
                # end if
            # end if
        # end for
    # end for

    return listOfMatchingCharsWithInnerCharRemoved


def distance_between_chars(first_char, second_char):
    # use Pythagorean theorem to calculate distance between two chars
    intX = abs(first_char.intCentreX - second_char.intCentreX)
    intY = abs(first_char.intCentreY - second_char.intCentreY)

    return math.sqrt((intX ** 2) + (intY ** 2))


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


def pre_processing(img):
    """
    The pre-processing of the image before it is feed to locating the character or directly to the recognition method
    """

    height, width, _ = img.shape

    # capture the necessary character information (the values were given as estimation the character would
    # occupy the region
    roi = img[int((height / 2) - (height / 2) * 0.90):int((height / 2) + (height / 2) * 0.90), int((width / 2) - (width
                                                                                                                  / 2) * 0.90):int(
        (width / 2) + (width / 2) * 0.90)]

    resize = cv2.resize(roi, (Settings.resize_height, Settings.resize_width), interpolation=cv2.INTER_AREA)

    # Convert the image to grayscale and turn to outline of the letter
    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)

    gauss = cv2.GaussianBlur(gray, (5, 5), 0)

    # if Step_letter:
    #   plt.hist(gauss.ravel(), bins=256, range=[0.0, 256.0], fc='k', ec='k')  # calculating histogram
    #   plt.show()

    if Settings.Step_letter:
        cv2.imshow("image", img)
        cv2.imshow("resize", resize)
        cv2.imshow("gray", gray)
        cv2.imshow("gauss", gauss)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # initialising variable
    preprocessedimage = None

    if Settings.preprocess_character == "otsu":
        if Settings.character == "tesseract":
            _, otsu = cv2.threshold(gauss, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            if Settings.Step_letter:
                cv2.imshow("otsu", otsu)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            preprocessedimage = otsu
        else:
            _, otsu = cv2.threshold(gauss, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if Settings.Step_letter:
                cv2.imshow("otsu", otsu)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            preprocessedimage = otsu

    elif Settings.preprocess_character == "custom":
        if Settings.character == "tesseract":
            kernel = np.ones((4, 4), np.uint8)
            edged = cv2.Canny(gauss, 10, 30)  # the lower the value the more detailed it would be
            dilate = cv2.dilate(edged, kernel, iterations=1)
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, kernel, iterations=1)
            close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=3)
            dilation = cv2.dilate(close, kernel, iterations=4)
            kernel = np.ones((4, 4), np.uint8)
            erode = cv2.erode(dilation, kernel, iterations=4)
            denoised = cv2.fastNlMeansDenoising(erode, None, 10, 7, 21)
            # filter image from grayscale to black and white
            img_thresh = cv2.adaptiveThreshold(denoised,  # input image
                                               255,  # make pixels that pass the threshold full white
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               # use gaussian rather than mean, seems to give better results
                                               cv2.THRESH_BINARY_INV,
                                               # invert so foreground will be white, background will be black
                                               11,  # size of a pixel neighborhood used to calculate threshold value
                                               0)  # constant subtracted from the mean or weighted mean
            thresh = img_thresh

        else:
            kernel = np.ones((4, 4), np.uint8)
            edged = cv2.Canny(gauss, 10, 30)  # the lower the value the more detailed it would be
            dilate = cv2.dilate(edged, kernel, iterations=1)
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, kernel, iterations=1)
            close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=3)
            dilation = cv2.dilate(close, kernel, iterations=4)
            kernel = np.ones((4, 4), np.uint8)
            erode = cv2.erode(dilation, kernel, iterations=4)
            denoised = cv2.fastNlMeansDenoising(erode, None, 10, 7, 21)
            thresh = denoised

        if Settings.Step_letter:
            cv2.imshow("edge", edged)
            cv2.imshow("dilate", dilate)
            cv2.imshow("open", opening)
            cv2.imshow("close", close)
            cv2.imshow("dilate2", dilation)
            cv2.imshow("erode", erode)
            cv2.imshow("denoised", denoised)
            cv2.imshow("thresh", thresh)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        preprocessedimage = thresh
    else:
        print("please chose between one of the preprocess method of 'otsu' or 'custom'.")

    return preprocessedimage


def area_of_region_of_interest(img_thresh_copy, distance, marker, i):
    # set heights and width to be able to read the image when comparing to flatten images
    h = 30
    w = 30
    allContoursWithData = []  # declare empty lists
    validContoursWithData = []  # we will fill these shortly
    npaROIResized = None
    directory = None

    newheight, newwidth = img_thresh_copy.shape

    (npaContours, _) = cv2.findContours(img_thresh_copy,
                                        # input image, make sure to use a copy since the function will modify
                                        # this image in the course of finding contours
                                        cv2.RETR_LIST,  # retrieve the outermost contours only
                                        cv2.CHAIN_APPROX_SIMPLE)  # compress horizontal, vertical, and diagonal
    # segments and leave only their end points

    if Settings.Step_letter:
        cv2.imshow("npaContours", img_thresh_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    for npaContour in npaContours:  # for each contour
        contourWithData = ContourWithData()  # instantiate a contour with data object
        contourWithData.npaContour = npaContour  # assign contour to contour with data
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)  # get the bounding rect
        contourWithData.bounding_rect_info()  # get bounding rect info
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)  # calculate the contour area
        allContoursWithData.append(contourWithData)  # add contour with data object to list of all contours with data
    # end for

    for contourWithData in allContoursWithData:  # for all contours
        if contourWithData.contour_valid(newheight, newwidth):  # check if valid
            validContoursWithData.append(contourWithData)  # if so, append to valid contour list
        # end if
    # end for

    validContoursWithData.sort(key=operator.attrgetter("intRectX"))  # sort contours from left to right
    validContoursWithData = remove_inner_overlapping_chars(validContoursWithData)  # removes overlapping letters

    for contourWithData in validContoursWithData:  # for each contour
        # new = cv2.cvtColor(cv2.rectangle(img,  # draw rectangle on original testing image
        #                                  (contourWithData.intRectX, contourWithData.intRectY),
        #                                  # upper left corner
        #                                  (contourWithData.intRectX + contourWithData.intRectWidth,
        #                                   contourWithData.intRectY + contourWithData.intRectHeight),
        #                                  # lower right corner
        #                                  (0, 255, 0),  # green
        #                                  2), cv2.COLOR_BGR2GRAY)  # thickness

        # crop char out of threshold image
        imgROI = img_thresh_copy[
                 contourWithData.intRectY + 1: contourWithData.intRectY + contourWithData.intRectHeight - 1,
                 contourWithData.intRectX + 1: contourWithData.intRectX + contourWithData.intRectWidth - 1]

        # resize image, this will be more consistent for recognition and storage
        imgROIResized = cv2.resize(imgROI, (w, h))

        if Settings.Save_Data:
            directory = "threshold"
        elif Settings.Static_Test:
            directory = "{0}".format(distance)
        elif Settings.Rover_Marker:
            directory = "marker={0}".format(marker)
        elif Settings.Distance_Test:
            directory = "{0}".format(Settings.dictory)

        write_to_file(directory, marker, i, "chosen", imgROIResized)

        npaROIResized = imgROIResized.reshape((1, w * h))  # flatten image into 1d numpy array

        npaROIResized = np.float32(
            npaROIResized)  # convert from 1d numpy array of ints to 1d numpy array of floats

        if Settings.Step_letter:
            cv2.imshow("resize", imgROIResized)
            # cv2.imshow("imgTestingNumbers", img) # show input image with green boxes drawn around found digits
            cv2.waitKey(0)
        # end if

    return npaROIResized
