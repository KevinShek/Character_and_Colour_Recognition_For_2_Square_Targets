import sys
import numpy as np
import cv2
import os

# module level variables ##########################################################################
MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 30
RESIZED_IMAGE_HEIGHT = 30

###################################################################################################
def main():
    imgTrainingNumbers = cv2.imread("classification.png")            # read in training numbers image

    if imgTrainingNumbers is None:                          # if image was not read successfully
        print("error: image not read from file \n\n")        # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return                                              # and exit function (which exits program)
    # end if

    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)          # get grayscale image
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                        # blur

    _, white = cv2.threshold(imgBlurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cv2.imshow("imgThresh", white)      # show threshold image for reference
    cv2.imshow("image", imgTrainingNumbers)      # show threshold image for reference

    imgThreshCopy = white.copy()        # make a copy of the thresh image, this in necessary b/c findContours modifies the image

    (npaContours,_) = cv2.findContours(imgThreshCopy,        # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                 cv2.RETR_EXTERNAL,                 # retrieve the outermost contours only
                                                 cv2.CHAIN_APPROX_SIMPLE)           # compress horizontal, vertical, and diagonal segments and leave only their end points

                                # declare empty numpy array, we will use this to write to file later
                                # zero rows, enough cols to hold all image data
    npaFlattenedImages = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
    no_rotate_flat = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

    intClassifications = []         # declare empty classifications list, this will be our list of how we are classifying our chars from user input, we will write to file at the end
    no_rotate = []         # declare empty classifications list, this will be our list of how we are classifying our chars from user input, we will write to file at the end

                                    # possible chars we are interested in are digits 0 through 9, put these in list intValidChars
    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]

    for npaContour in npaContours:                          # for each contour
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:          # if contour is big enough to consider
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)         # get and break out bounding rect

                                                # draw rectangle around each contour as we ask user for input
            cv2.rectangle(imgTrainingNumbers,           # draw rectangle on original training image
                          (intX, intY),                 # upper left corner
                          (intX+intW,intY+intH),        # lower right corner
                          (0, 0, 255),                  # red
                          2)                            # thickness

            imgROI = white[intY:intY+intH, intX:intX+intW]                                  # crop char out of threshold image
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))     # resize image, this will be more consistent for recognition and storage

            ###########
            rotate = cv2.getRotationMatrix2D((RESIZED_IMAGE_WIDTH/2, RESIZED_IMAGE_HEIGHT/2), 180, 1.0)
            rotate180 = cv2.warpAffine(imgROIResized, rotate, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
            _, black = cv2.threshold(imgROIResized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            cv2.imshow("imgROI", imgROI)                    # show cropped out char for reference
            cv2.imshow("imgROIResized", imgROIResized)      # show resized image for reference
            # cv2.imshow("90", rotate90)
            cv2.imshow("180", rotate180)
            cv2.imshow("black", black)

            cv2.imshow("training_numbers.png", imgTrainingNumbers)      # show training numbers image, this will now have red rectangles drawn on it

            intChar = cv2.waitKey(0)                     # get key press

            if intChar == 27:                   # if esc key was pressed
                sys.exit()                      # exit program
            elif intChar in intValidChars:      # else if the char is in the list of chars we are looking for . . .

                intClassifications.append(intChar)                                                # append classification char to integer list of chars (we will convert to float later before writing to file)

                npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # flatten image to 1d numpy array so we can write to file later
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)                    # add current flattened impage numpy array to list of flattened image numpy arrays
                intClassifications.append(intChar)  # append classification char to integer list of chars (we will convert to float later before writing to file)
                npaFlattenedImage = rotate180.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # flatten image to 1d numpy array so we can write to file later
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)                    # add current flattened impage numpy array to list of flattened image numpy arrays
                intClassifications.append(intChar)  # append classification char to integer list of chars (we will convert to float later before writing to file)
                npaFlattenedImage = black.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # flatten image to 1d numpy array so we can write to file later
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)                    # add current flattened impage numpy array to list of flattened image numpy arrays

                no_rotate.append(intChar)                                                # append classification char to integer list of chars (we will convert to float later before writing to file)
                npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # flatten image to 1d numpy array so we can write to file later
                no_rotate_flat = np.append(no_rotate_flat, npaFlattenedImage, 0)                    # add current flattened impage numpy array to list of flattened image numpy arrays

                # for i in range(90, 360, 90):
                #   if i == 90:
                #     npaFlattenedImage = rotate90.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                #     npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)
                #   elif i == 180:
                #     npaFlattenedImage = rotate180.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                #     npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)
                #   elif i == 270:
                #     npaFlattenedImage = rotate270.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                #     npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)
            # end if
        # end if
    # end for

    fltClassifications = np.array(intClassifications, np.float32)                   # convert classifications list of ints to numpy array of floats

    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))   # flatten numpy array of floats to 1d so we can write to file later

    no_rotate_list = np.array(no_rotate, np.float32)                   # convert classifications list of ints to numpy array of floats
    flattenrotate = no_rotate_list.reshape((no_rotate_list.size, 1))   # flatten numpy array of floats to 1d so we can write to file later


    print ("\n\ntraining complete !!\n")

    np.savetxt("classwithi.txt", npaClassifications)           # write flattened images to file
    np.savetxt("flatwithi.txt", npaFlattenedImages)
    np.savetxt("classification_no_rotate.txt", flattenrotate)           # write flattened images to file
    np.savetxt("flat_no_rotate.txt", no_rotate_flat)

    cv2.destroyAllWindows()             # remove windows from memory

    return

###################################################################################################
if __name__ == "__main__":
    main()
# end if



