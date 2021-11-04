import os
import cv2
import numpy as np


class Settings:
    def __init__(self):
        # The steps of the programs
        self.Step_camera = False  # stages of camera activating
        self.Step_letter = False  # view the stages of character recognition
        self.Step_color = False  # view the stages of colour recognition
        self.Step_detection = False  # stages of detection

        # Saving the data depending on the situation, only one of the 3 settings can be set true at a time
        self.save_results = True  # to allow saving to occur
        self.name_of_folder = "exp" # name of the folder that could be found in "result_dir/exp"
        self.exist_ok = False # it would save the content into the same name of folder if it exist within result_dir

        # Provide a video or dataset you wish to test against
        self.media = "test_images"  # video used for testing and has to be in the exact location of the config file
        # video/target_only.mp4
        # Information
        self.capture = "pi"  # "pc" to work with a PC and "pi" to work for a raspberry pi or "image" for a single frame
        # capture
        # detection and recognition
        self.testing = "video"  # are you running the program for a "video" to use this capture has to be "pc" or "pi"
        # or testing the "detection" this includes recognition or "detection_only" or "character" or "colour" recognition
        # capture must be "image"
        self.pause = False # captures images until it hits the counter for webcam

        # Testing the recognition to enable this capture has to be "image" and testing = "character" or "colour",
        # one at a time.
        self.character_test = False  # character recognition test
        self.colour_test = False  # colour recognition test

        # Methods
        self.device_for_tesseract = "pi"
        self.character = "tesseract"  # for character recognition there is currently 2 setting "knn" or "tesseract"
        self.knn_value = 3  # the knn value used for knn process (only odd positive number works)
        self.preprocess_character = "otsu"  # this is the threshold before it is feed into character recognition method currently
        # there is "otsu" or "custom"
        self.colour = "rgb"  # for colour recognition there is currently 2 setting "rgb" or "hsv"
        self.preprocess_color = ""  # the pre processing on normalising the colour by the use of character as
        # an anchor for actual white there are 4 options, "rgb_difference", "hsv_difference", "temperature_colour", " "

        # There is 3 option 1, 2, 3. 1 is for the inner square only, 2 is for detecting the outer and inner square,
        # then 3 is detecting the outer square and force cropping to retrieve the inner square.
        self.square = 2

        # character recognition for K-NN
        # Load training and classification data
        npaClassifications = np.loadtxt("classifications_no_rotate.txt", np.float32)
        self.npaFlattenedImages = np.loadtxt("flatten_no_rotate.txt", np.float32)

        # reshape classifications array to 1D for k-nn
        self.Classifications = npaClassifications.reshape((npaClassifications.size, 1))

        # values for temperature colour correction method
        self.kelvin_table = {
            1000: (255, 56, 0),
            1100: (255, 71, 0),
            1200: (255, 83, 0),
            1300: (255, 93, 0),
            1400: (255, 101, 0),
            1500: (255, 109, 0),
            1600: (255, 115, 0),
            1700: (255, 121, 0),
            1800: (255, 126, 0),
            1900: (255, 131, 0),
            2000: (255, 138, 18),
            2100: (255, 142, 33),
            2200: (255, 147, 44),
            2300: (255, 152, 54),
            2400: (255, 157, 63),
            2500: (255, 161, 72),
            2600: (255, 165, 79),
            2700: (255, 169, 87),
            2800: (255, 173, 94),
            2900: (255, 177, 101),
            3000: (255, 180, 107),
            3100: (255, 184, 114),
            3200: (255, 187, 120),
            3300: (255, 190, 126),
            3400: (255, 193, 132),
            3500: (255, 196, 137),
            3600: (255, 199, 143),
            3700: (255, 201, 148),
            3800: (255, 204, 153),
            3900: (255, 206, 159),
            4000: (255, 209, 163),
            4100: (255, 211, 168),
            4200: (255, 213, 173),
            4300: (255, 215, 177),
            4400: (255, 217, 182),
            4500: (255, 219, 186),
            4600: (255, 221, 190),
            4700: (255, 223, 194),
            4800: (255, 225, 198),
            4900: (255, 227, 202),
            5000: (255, 228, 206),
            5100: (255, 230, 210),
            5200: (255, 232, 213),
            5300: (255, 233, 217),
            5400: (255, 235, 220),
            5500: (255, 236, 224),
            5600: (255, 238, 227),
            5700: (255, 239, 230),
            5800: (255, 240, 233),
            5900: (255, 242, 236),
            6000: (255, 243, 239),
            6100: (255, 244, 242),
            6200: (255, 245, 245),
            6300: (255, 246, 247),
            6400: (255, 248, 251),
            6500: (255, 249, 253),
            6600: (254, 249, 255),
            6700: (252, 247, 255),
            6800: (249, 246, 255),
            6900: (247, 245, 255),
            7000: (245, 243, 255),
            7100: (243, 242, 255),
            7200: (240, 241, 255),
            7300: (239, 240, 255),
            7400: (237, 239, 255),
            7500: (235, 238, 255),
            7600: (233, 237, 255),
            7700: (231, 236, 255),
            7800: (230, 235, 255),
            7900: (228, 234, 255),
            8000: (227, 233, 255),
            8100: (225, 232, 255),
            8200: (224, 231, 255),
            8300: (222, 230, 255),
            8400: (221, 230, 255),
            8500: (220, 229, 255),
            8600: (218, 229, 255),
            8700: (217, 227, 255),
            8800: (216, 227, 255),
            8900: (215, 226, 255),
            9000: (214, 225, 255),
            9100: (212, 225, 255),
            9200: (211, 224, 255),
            9300: (210, 223, 255),
            9400: (209, 223, 255),
            9500: (208, 222, 255),
            9600: (207, 221, 255),
            9700: (207, 221, 255),
            9800: (206, 220, 255),
            9900: (205, 220, 255),
            10000: (207, 218, 255),
            10100: (207, 218, 255),
            10200: (206, 217, 255),
            10300: (205, 217, 255),
            10400: (204, 216, 255),
            10500: (204, 216, 255),
            10600: (203, 215, 255),
            10700: (202, 215, 255),
            10800: (202, 214, 255),
            10900: (201, 214, 255),
            11000: (200, 213, 255),
            11100: (200, 213, 255),
            11200: (199, 212, 255),
            11300: (198, 212, 255),
            11400: (198, 212, 255),
            11500: (197, 211, 255),
            11600: (197, 211, 255),
            11700: (197, 210, 255),
            11800: (196, 210, 255),
            11900: (195, 210, 255),
            12000: (195, 209, 255)}

        self.kelvin_list = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900,
                    2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900,
                    3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900,
                    4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900,
                    5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900,
                    6000, 6100, 6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900,
                    7000, 7100, 7200, 7300, 7400, 7500, 7600, 7700, 7800, 7900,
                    8000, 8100, 8200, 8300, 8400, 8500, 8600, 8700, 8800, 8900,
                    9000, 9100, 9200, 9300, 9400, 9500, 9600, 9700, 9800, 9900,
                    10000, 10100, 10200, 10300, 10400, 10500, 10600, 10700, 10800, 10900,
                    11000, 11100, 11200, 11300, 11400, 11500, 11600, 11700, 11800, 11900,
                    12000]

