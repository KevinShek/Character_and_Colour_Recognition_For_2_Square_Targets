import cv2
import numpy as np
from collections import Counter
import webcolors
import os
from rgb_discrete_dict import rgb_dict
from config import Settings
from PIL import Image

"""
The recognition of the inner square colour
"""


def colour(img):
    results = []  # empty list
    general_name = None
    config = Settings()

    # the following resize height and width are used for the resizing of the images before pre-processing occurs
    resize_height = 100
    resize_width = 100

    nroi = cv2.resize(img, (resize_height, resize_width), interpolation=cv2.INTER_AREA)

    new_resize = pre_processing(nroi, config, resize_height, resize_width)

    h = 30
    w = 30

    resizeBGR = cv2.resize(new_resize, (w, h))  # reduces the size of the image so the process would run fast

    if config.Step_color:
        cv2.imshow("image", img)
        cv2.imshow("resizeBGR", resizeBGR)
        cv2.waitKey(0)

    if config.preprocess_color == "hsv_difference":
        resize_hsv = np.uint8(new_resize)
        new_resize = cv2.cvtColor(resize_hsv, cv2.COLOR_HSV2RGB)

    if config.colour == "rgb":
        results.append(rgb_colour(resizeBGR, config).most_common(1)[0][0])

    elif config.colour == "hsv":
        results.append(hsv_colour(resizeBGR, config))

    mode = Counter(results)

    if config.Step_color:
        print(mode)

    if mode == Counter():
        general_name = "None"
    else:
        if config.colour == "rgb":
            colourname = mode.most_common(1)[0][0]
            general_name = rgb_dict(colourname)
        elif config.colour == "hsv":
            general_name = mode.most_common(1)[0][0]

    return general_name, new_resize


def pre_processing(resize, config, resize_height, resize_width):
    # Convert the image to grayscale and turn to outline of the letter
    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(gauss, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    number_of_white_pixel = (resize_height * resize_width) - cv2.countNonZero(otsu)
    otsu_converted = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
    anchor_image = cv2.subtract(resize, otsu_converted)
    white_image = np.ones((resize_height, resize_width, 3)) * 255  # creating pure white image
    if config.Step_color:
        cv2.imshow("resize", resize)
        cv2.imshow("otsu", otsu)
        cv2.imshow("anchor_image", anchor_image)
        cv2.imshow("white_image", white_image)
        print(number_of_white_pixel)
        cv2.waitKey(0)
    if config.preprocess_color == "rgb_difference":
        difference = np.subtract(white_image[:, :, :], anchor_image[:, :, :])  # subtracting the pure white image by the
        # image of the letter to get the difference
        actual_difference = np.where(difference == 255, 0, difference)  # replace all 255 to 0 (away to rid the
        # values that is not of interest)
        mean1 = np.sum(actual_difference[:, :, 0])/number_of_white_pixel  # mean for the blue
        mean2 = np.sum(actual_difference[:, :, 1])/number_of_white_pixel  # mean for the green
        mean3 = np.sum(actual_difference[:, :, 2])/number_of_white_pixel  # mean for the red
        # new_resize2 = np.zeros((resize_height, resize_width, 3))
        new_resize = np.add(resize[:, :, :], [mean1, mean2, mean3])  # adding the means to their respective bgr
        # new_resize = new_resize/255
        # new_resize2 = np.add(resize[1, :, :], mean1)
        # new_resize2 = np.add(resize[:, 1, :], mean2)
        # new_resize2 = np.add(resize[:, :, 1], mean3)
        # new_resize[:, 1, :] = cv2.add(resize[:, 1, :], [mean1, mean2, mean3])
        # new_resize[:, 1, :] = cv2.add(resize[:, 1, :], [mean1, mean2, mean3])
        new_resize = np.where(new_resize > 255, 255, new_resize)  # replace all over 255 to 255
        new_resize = np.uint8(new_resize)  # convert the values from a float to an int8 for imshow
        processed = new_resize
        if config.Step_color:
            cv2.imshow("differenecee", difference)
            cv2.imshow("actual_difference", actual_difference)
            print("resize before = {0}".format(resize))
            print("resize after = {0}".format(new_resize))
            print(new_resize.shape)
            print(resize.shape)
            cv2.imshow("new resize", new_resize)
            cv2.waitKey(0)

    elif config.preprocess_color == "hsv_difference":
        white_image_hsv = np.float32(white_image)
        anchor_image_hsv = np.float32(anchor_image)
        resize_hsv = np.float32(resize)
        white_image_hsv = cv2.cvtColor(white_image_hsv, cv2.COLOR_BGR2HSV_FULL)
        anchor_hsv = cv2.cvtColor(anchor_image_hsv, cv2.COLOR_BGR2HSV_FULL)
        resize_hsv = cv2.cvtColor(resize_hsv, cv2.COLOR_BGR2HSV_FULL)
        white_image_hsv[:, :, 1] = np.dot(white_image_hsv[:, :, 1], 255)
        anchor_hsv[:, :, 1] = np.dot(anchor_hsv[:, :, 1], 255)
        resize_hsv[:, :, 1] = np.dot(resize_hsv[:, :, 1], 255)
        difference = np.subtract(white_image_hsv[:, :, :], anchor_hsv[:, :, :])  # subtracting the pure white image
        # by the image of the letter to get the difference
        difference[:, :, 2] = np.where(difference[:, :, 2] == 255, 0, difference[:, :, 2])  # replace all 255 to 0 (away to rid the
        # values that is not of interest)
        mean2 = (np.sum(abs(difference[:, :, 1]))/number_of_white_pixel) * -1  # mean for the saturation
        mean3 = np.sum(difference[:, :, 2])/number_of_white_pixel  # mean for the value
        new_resize = np.add(resize_hsv[:, :, :], [0, mean2, mean3])  # adding the means to their respective hsv
        new_resize[:, :, 1] = np.where(new_resize[:, :, 1] < 0, 0, new_resize[:, :, 1])  # replace all over 255 to 255
        new_resize[:, :, 2] = np.where(new_resize[:, :, 2] > 255, 255, new_resize[:, :, 2])  # replace all over 255
        # to 255

        processed = new_resize

        if config.Step_color:
            cv2.imshow("differenecee", difference)
            # print("white_image_hsv = {0}".format(white_image_hsv))
            # print("anchor_hsv = {0}".format(anchor_hsv))
            # print("resize before = {0}".format(resize_hsv))
            print("resize after = {0}".format(new_resize))
            cv2.imshow("acutal_differenecee", difference)
            new_resize = np.uint8(new_resize)  # convert the values from a float to an int8 for imshow
            cv2.imshow("new resize", new_resize)
            new_resize_rgb = cv2.cvtColor(new_resize, cv2.COLOR_HSV2BGR_FULL)
            new_resize_rgb = np.uint8(new_resize_rgb)
            cv2.imshow("new resize rgb", new_resize_rgb)
            cv2.waitKey(0)
    elif config.preprocess_color == "temperature_colour":
        _, inverse_otsu = cv2.threshold(gauss, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inverse_otsu_converted = cv2.cvtColor(inverse_otsu, cv2.COLOR_GRAY2BGR)
        list_of_result = [] * len(config.kelvin_list)
        for i in range(0, len(config.kelvin_list)):
            b, g, r = cv2.split(anchor_image)
            kevin_value = config.kelvin_list[i]
            temp = config.kelvin_table[kevin_value]
            r_temp, g_temp, b_temp = temp
            r = cv2.addWeighted(src1=r, alpha=r_temp/255.0, src2=0, beta=0, gamma=0)
            g = cv2.addWeighted(src1=g, alpha=g_temp/255.0, src2=0, beta=0, gamma=0)
            b = cv2.addWeighted(src1=b, alpha=b_temp/255.0, src2=0, beta=0, gamma=0)
            balance_img = cv2.merge([b, g, r])
            greyscale = cv2.cvtColor(balance_img, cv2.COLOR_BGR2GRAY)
            difference = np.subtract(inverse_otsu[:, :], greyscale[:, :])
            mean = np.sum(difference[:, :]) / number_of_white_pixel
            # difference = np.subtract(inverse_otsu_converted[:, :, :], balance_img[:, :, :])
            # mean = (np.sum(difference[:, :, 0]) / number_of_white_pixel + np.sum(difference[:, :, 1]) /
            #         number_of_white_pixel + np.sum(difference[:, :, 2]) / number_of_white_pixel) / 3
            list_of_result.append(mean)

        position = np.argmin(list_of_result)
        b, g, r = cv2.split(resize)
        kevin_value = config.kelvin_list[position]
        temp = config.kelvin_table[kevin_value]
        r_temp, g_temp, b_temp = temp
        r = cv2.addWeighted(src1=r, alpha=r_temp / 255.0, src2=0, beta=0, gamma=0)
        g = cv2.addWeighted(src1=g, alpha=g_temp / 255.0, src2=0, beta=0, gamma=0)
        b = cv2.addWeighted(src1=b, alpha=b_temp / 255.0, src2=0, beta=0, gamma=0)
        balance_img = cv2.merge([b, g, r])
        processed = balance_img

        if config.Step_color:
            print(position)
            print(list_of_result[position])
            cv2.imshow("balance_image", processed)
            cv2.imshow("inverse_image", inverse_otsu_converted)
            print(inverse_otsu_converted)
            cv2.waitKey(0)

    else:
        processed = resize
    return processed


def hsv_colour(resizeBGR, config):
    if config.preprocess_color == "hsv_difference":
        resizeHSV = resizeBGR
    else:
        resizeBGR = np.float32(resizeBGR)
        resizeHSV = cv2.cvtColor(resizeBGR, cv2.COLOR_BGR2HSV_FULL)
        resizeHSV[:, :, 1] = np.dot(resizeHSV[:, :, 1], 255)

    if config.Step_color:
        cv2.imshow("resize", resizeHSV)
        cv2.waitKey(0)

    # Colour Boundaries red, blue, yellow and gray HSV
    # this requires dtype="uint16" for lower and upper & HSV = np.float32(HSV) before the conversion of HSV_FULL
    boundaries = [("black", [0, 0, 0], [360, 255, 31.875]), ("white", [0, 0, 223.125], [360, 63.75, 255]),
                  ("yellow-red", [15, 63.75, 31.875], [45, 255, 255]),
                  ("yellow", [45, 63.75, 31.875], [75, 255, 255]),
                  ("yellow green", [75, 63.75, 31.875], [105, 255, 255]),
                  ("green", [105, 63.75, 31.875], [135, 255, 255]),
                  ("green cyan", [135, 63.75, 31.875], [165, 255, 255]),
                  ("cyan", [165, 63.75, 31.875], [195, 255, 255]),
                  ("blue cyan", [195, 63.75, 31.875], [225, 255, 255]),
                  ("blue", [225, 63.75, 31.875], [255, 255, 255]),
                  ("blue magenta", [255, 63.75, 31.875], [285, 255, 255]),
                  ("magenta", [285, 63.75, 31.875], [315, 255, 255]),
                  ("red magenta", [315, 63.75, 31.875], [345, 255, 255]),
                  ("red", [0, 63.75, 31.875], [15, 255, 255]),
                  ("red", [345, 63.75, 31.875], [360, 255, 255]),
                  ("gray", [0, 0, 31.875], [360, 63.75, 223.125])]

    # initialising variables
    ratio = [] * len(boundaries)
    if config.Step_color:
        i = 0

    # comparing each pixel of the picture and append the colour name in to a list (BGR to RGB to get the name)
    for (color, lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint16")
        upper = np.array(upper, dtype="uint16")

        mask = cv2.inRange(resizeHSV, lower, upper)

        ratio.append(np.round((cv2.countNonZero(mask) / (resizeHSV.size / 3)) * 100, 2))

        if config.Step_color:
            print(color, ratio[i])
            cv2.imshow("mask", mask)
            i = i + 1
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # locating the highest value in an array retrieving tis position in the array
    position = np.argmax(ratio)

    return boundaries[position][0]


def rgb_colour(resizeBGR, config):
    if config.preprocess_color == "hsv_difference":
        resizeBGR = np.float32(resizeBGR)
        resizeRGB = cv2.cvtColor(resizeBGR, cv2.COLOR_HSV2RGB)
    else:
        resizeRGB = cv2.cvtColor(resizeBGR, cv2.COLOR_BGR2RGB)

    if config.Step_color:
        if config.preprocess_color == "hsv_difference":
            resizeBGR = cv2.cvtColor(resizeBGR, cv2.COLOR_HSV2BGR)
            cv2.imshow("resize", resizeBGR)
            cv2.waitKey(0)
        else:
            cv2.imshow("resize", resizeRGB)
            cv2.waitKey(0)

    # Flatten image while keeping vectors intact (i.e. list of vectors)
    resizeRGB = np.reshape(resizeRGB, (-1, 3))

    # Get the RGB values of the colors in 'Extended colors' dataset (CSS3)
    colours = np.array(list(list(webcolors.hex_to_rgb(key)) for key, _ in webcolors.CSS3_HEX_TO_NAMES.items()))

    # Colour descriptors of the colors in CSS3 (hsv,name)
    names = list(webcolors.CSS3_HEX_TO_NAMES.items())

    # Initialise the name list
    rgb_name_storage = [None] * resizeRGB.shape[0]

    # Loop through the pixels in the image
    for i, pixel in enumerate(resizeRGB):
        # Get Euclidean distance between the pixel and the CSS3 colors in a 3D space where R,G,B are the
        # dimensions
        distances = np.linalg.norm((colours - pixel), axis=1)
        # Get index of closest colour to the pixel's colour
        closest_idx = np.argmin(distances)

        # Append name of colour to list , [0] gives you the HEX value
        rgb_name_storage[i] = names[closest_idx][1]

    # Get the frequency of each color name in the list
    return Counter(rgb_name_storage)
