import os
import cv2


class Settings:
    # Vehicle used and GPS switch
    GPS = False  # GPS
    air = 0  # setting to see if it is ground base or not

    # The steps of the programs
    Step_camera = False  # stages of camera activating
    Step_letter = True  # view the stages of character recognition
    Step_color = True  # view the stages of colour recognition
    Step_detection = False  # stages of detection

    # Saving the data depending on the situation, only one of the 3 settings can be set true at a time
    Static_Test = False  # for distance test
    Rover_Marker = False  # saves images into their individual marker files
    Save_Data = False  # Saving Data into files
    Distance_Test = False

    # local path
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

    # This is information for testing a single frame
    rel_path = "Test_Images/85_6.png"  # where is the image is located from where the config is
    # current located
    abs_file_path = os.path.join(script_dir, rel_path)  # attaching the location
    test_image = cv2.imread(abs_file_path)  # reading in the image

    # Distance saving file
    counter = 0
    switch = True
    dictory = None
    number = None
    alphanumeric_character = None
    file_path = "8M"

    video = "new.mp4"  # video used for testing and has to be in the exact location of the config file

    # Information
    capture = "image"  # "pc" to work with a PC and "pi" to work for a raspberry pi or "image" for a single frame
    # capture
    # detection and recognition
    testing = "colour"  # are you running the program for a "video" to use this capture has to be "pc" or "pi"
    # or testing the "detection" this includes recognition or "character" or "colour" recognition capture must be
    # "image"

    # Testing the recognition to enable this capture has to be "image" and testing = "character" or "colour",
    # one at a time.
    character_test = False  # character recognition test
    colour_test = True  # colour recognition test

    # Methods
    character = "tesseract"  # for character recognition there is currently 2 setting "knn" or "tesseract"
    knn_value = 3  # the knn value used for knn process (only odd positive number works)
    preprocess_character = "otsu"  # this is the threshold before it is feed into character recognition method currently
    # there is "otsu" or "custom"
    colour = "hsv"  # for colour recognition there is currently 2 setting "rgb" or "hsv"
    preprocess_color = ""  # the pre processing on normalising the colour by the use of character as an
    # anchor for actual white there are 3 options, "rgb_difference", "hsv_difference", or "" to normalise the colour

    # the following resize height and width are used for the resizing of the images before pre-processing occurs
    resize_height = 100
    resize_width = 100

    # There is 3 option 1, 2, 3. 1 is for the inner square only, 2 is for detecting the outer and inner square,
    # then 3 is detecting the outer square and force cropping to retrieve the inner square.
    square = 2
