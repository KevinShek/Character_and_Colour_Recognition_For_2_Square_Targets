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
from detection import detection

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
    config = Settings()
    if config.capture != "image":
        marker += 1

    return marker, counter


def results_of_distance_test(filename, predicted_character, predicted_color, actual_character, actual_color, result_dir, resolution_used, internal_folder):
    if predicted_character == actual_character:
        correct_character = 1
    else:
        correct_character = 0
    if predicted_color == actual_color:
        correct_colour = 1
    else:
        correct_colour = 0

    with open(f'{result_dir}/{resolution_used}_results.csv', 'a') as csvfile:  # for testing purposes
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        filewriter.writerow([str(filename), str(predicted_character), str(actual_character), str(correct_character), str(predicted_color), str(actual_color), str(correct_colour)])


def results_of_static_test(filename, predicted_character, predicted_color, actual_character, actual_color, result_dir, config):
    if predicted_character == actual_character:
        correct_character = 1
    else:
        correct_character = 0
    if predicted_color == actual_color:
        correct_colour = 1
    else:
        correct_colour = 0

    with open(f'{result_dir}/{config.name_of_folder}_results.csv', 'a') as csvfile:  # for testing purposes
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        filewriter.writerow([str(filename), str(predicted_character), str(actual_character), str(correct_character), str(predicted_color), str(actual_color), str(correct_colour)])


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
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)  # 800 default
            cap.set(3, config.width)  # 800 default
            cap.set(4, config.height)  # 800 default
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
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
        camera.resolution = (config.width, config.height)
        camera.brightness = 50  # 50 is default
        camera.framerate = 90
        camera.awb_mode = 'auto'
        camera.shutter_speed = camera.exposure_speed
        cap = PiRGBArray(camera, size=(config.width, config.height))

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
            # clear the stream in preparation for the next frame
            cap.truncate(0)
                
    elif config.capture == "image":
        if config.testing == "none":
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

        if config.testing == "distance_test":
            data_dir = Path(config.media)
            resolution_used = f"{config.media}".rsplit('/', 1)[-1] # grabbing the name of the folder name

            with open(f'{save.save_dir}/{resolution_used}_results.csv', 'a') as csvfile:  # making the csv file
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                filewriter.writerow(["Filename" , "Predicted Character", "Actual Character", "Correct Character", "Predicted Colour", "Actual Colour", "Correct Colour"])
            with open(f'{save.save_dir}/{resolution_used}_results_overall.csv', 'a') as csvfile:  # for testing purposes
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                filewriter.writerow(["Distance", "Correct Guess Character", "Correct Guess Character (%)", "Correct Guess Colour", "Correct Guess Colour (%)", "Detection Speed (ms)", "Character Recognition Speed (ms)", "Colour Recognition Speed (ms)", "Total Speed (ms)", f"Character Regnition Method ({config.character})", f"Colour Regnition Method ({config.colour})"])

            list_of_internal_folders = sorted(os.listdir(data_dir)) # sorted into numerical order

            for internal_folder in list_of_internal_folders:
                cap = [] # to store the names of the images
                correct_prediction_of_character, correct_prediction_of_colour= 0, 0
                t0, t1, t2 = 0., 0., 0.
                seen = 0

                # making internal folders for each distances
                Path(f"{save.save_dir}/{internal_folder}").mkdir(parents=True, exist_ok=True)

                data_dir = Path(f"{config.media}/{internal_folder}") # path to the images where it is located
                # the following code interite over the extension that exist within a folder and place them into a single list
                image_count = sorted(list(itertools.chain.from_iterable(data_dir.glob(pattern) for pattern in ('*.jpg', '*.png'))))
                # image_count = len(list(data_dir.glob('*.jpg')))
                for name in image_count:
                    # head, tail = ntpath.split(name)
                    filename = Path(name)  # .stem removes the extension and .name grabs the filename with extension
                    cap.append(filename) # append the path of the images to the list
                    name_of_image = filename.stem
                    # print(name_of_image)
                    actual_character = f"{name_of_image}".rsplit('_', 2)[0]

                    if config.colour == "hsv":
                        dict_of_colour_with_character = {'1': "gray", '2': "gray", '3': "black", '4': "black", '5': "cyan", '6': "cyan", '7': "gray", '8': "gray", '9': "yellow", 
                        'A': "cyan", 'B': "red", 'C': "yellow-red", 'D': "blue cyan", 'E': "blue", 'F': "blue cyan", 'G': "yellow", 'H': "yellow", 'I': "green",'J': "yellow", 
                        'K': "yellow", 'L': "magenta", 'M': "magenta", 'N': "magenta", 'O': "green", 'P': "green", 'Q': "red", 'R': "blue", 'S': "cyan", 
                        'T': "blue", 'U': "green", 'V': "red", 'W': "red", 'X': "yellow-red", 'Y': "blue", 'Z': "magenta"}
                    elif config.colour == "rgb":
                        dict_of_colour_with_character = {'1': "grey", '2': "grey", '3': "grey", '4': "grey", '5': "cyan", '6': "cyan", '7': "grey", '8': "grey", '9': "green", 
                        'A': "cyan", 'B': "brown", 'C': "orange", 'D': "blue", 'E': "blue", 'F': "blue", 'G': "yellow", 'H': "yellow", 'I': "green",'J': "green", 
                        'K': "green", 'L': "magenta", 'M': "magenta", 'N': "magenta", 'O': "green", 'P': "green", 'Q': "red", 'R': "blue", 'S': "cyan", 
                        'T': "blue", 'U': "green", 'V': "brown", 'W': "red", 'X': "orange", 'Y': "blue", 'Z': "magenta"}
                    else:
                        print("please choose hsv or rgb")
                        break

                    list_of_character = list(dict_of_colour_with_character.keys())
                    list_of_colour = list(dict_of_colour_with_character.values())

                    actual_colour = list_of_colour[list_of_character.index(str(actual_character))]

                    test_image = cv2.imread(str(filename))

                    t = time.time()
                    color, roi, frame, success = detection(test_image, config)

                    if success:
                        t0 += time.time() - t
                        t = time.time()
                        predicted_character, contour_image, chosen_image = character_recognition.character(color)
                        t1 += time.time() - t
                        t = time.time()
                        predicted_color, processed_image = colour_recognition.colour(color)
                        t2 += time.time() - t

                        seen += 1
    
                        if predicted_character == actual_character:
                            correct_prediction_of_character += 1
                        if predicted_color == actual_colour:
                            correct_prediction_of_colour += 1
                        
                        results_of_distance_test(name_of_image, predicted_character, predicted_color, actual_character, actual_colour, save.save_dir, resolution_used, internal_folder)
                        
                        if config.save_results:
                            name_of_results = ["color", "roi", "frame","contour_image","processed_image", "chosen_image"]
                            image_results = [color, roi, frame, contour_image, processed_image, chosen_image]
                            for value, data in enumerate(name_of_results):
                                image_name = f"{internal_folder}/{name_of_image}_{data}.jpg"
                                image = image_results[value]
                                if image is not None:
                                    save.save_the_image(image_name, image)

                percentage_of_correct_character = (correct_prediction_of_character/len(cap)) * 100
                percentage_of_correct_colour = (correct_prediction_of_colour/len(cap)) * 100

                if seen != 0: 
                    speed = tuple(x / seen * 1E3 for x in (t0, t1, t2, t0 + t1 + t2))
                else:
                    speed = [0.0, 0.0, 0.0, 0.0]

                with open(f'{save.save_dir}/{resolution_used}_results_overall.csv', 'a') as csvfile:  # for testing purposes
                    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                    filewriter.writerow([str(internal_folder), str(correct_prediction_of_character), str(percentage_of_correct_character), str(correct_prediction_of_colour), str(percentage_of_correct_colour), str(speed[0]), str(speed[1]), str(speed[2]), str(speed[3])])

                print(f"percentage_of_correct_character = {percentage_of_correct_character}")
                print(f"percentage_of_correct_colour = {percentage_of_correct_colour}")

            with open(f'{save.save_dir}/{resolution_used}_information.txt', 'a') as f:
                f.write("Colour and Character Recognition for a 2 Square Target \n")
                f.write(f"Resolution of the images = {resolution_used} \n")
                f.write(f"method used for character recognition = {config.character} \n")
                f.write(f"method used for colour recognition = {config.colour} \n")
                f.write("\n")
        
        if config.testing == "static_test":
            cap = [] # to store the names of the images
            correct_prediction_of_character, correct_prediction_of_colour= 0, 0
            t0, t1, t2 = 0., 0., 0.
            seen = 0
            data_dir = Path(config.media)

            with open(f'{save.save_dir}/{config.name_of_folder}_results.csv', 'a') as csvfile:  # making the csv file
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                filewriter.writerow(["Filename" , "Predicted Character", "Actual Character", "Correct Character", "Predicted Colour", "Actual Colour", "Correct Colour"])
            with open(f'{save.save_dir}/{config.name_of_folder}_results_overall.csv', 'a') as csvfile:  # for testing purposes
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                filewriter.writerow(["Distance", "Correct Guess Character", "Correct Guess Character (%)", "Correct Guess Colour", "Correct Guess Colour (%)", "Detection Speed (ms)", "Character Recognition Speed (ms)", "Colour Recognition Speed (ms)", "Total Speed (ms)", f"Character Regnition Method ({config.character})", f"Colour Regnition Method ({config.colour})"])

            # the following code interite over the extension that exist within a folder and place them into a single list
            image_count = sorted(list(itertools.chain.from_iterable(data_dir.glob(pattern) for pattern in ('*.jpg', '*.png'))))
            # image_count = len(list(data_dir.glob('*.jpg')))
            for name in image_count:
                # head, tail = ntpath.split(name)
                filename = Path(name)  # .stem removes the extension and .name grabs the filename with extension
                cap.append(filename) # append the path of the images to the list
                name_of_image = filename.stem
                # print(name_of_image)
                actual_character = f"{name_of_image}".rsplit('_', 1)[0]

                if config.colour == "hsv":
                    dict_of_colour_with_character = {'1': "gray", '2': "gray", '3': "black", '4': "black", '5': "cyan", '6': "cyan", '7': "gray", '8': "gray", '9': "yellow", 
                    'A': "cyan", 'B': "red", 'C': "yellow-red", 'D': "blue cyan", 'E': "blue", 'F': "blue cyan", 'G': "yellow", 'H': "yellow", 'I': "green",'J': "yellow", 
                    'K': "yellow", 'L': "magenta", 'M': "magenta", 'N': "magenta", 'O': "green", 'P': "green", 'Q': "red", 'R': "blue", 'S': "cyan", 
                    'T': "blue", 'U': "green", 'V': "red", 'W': "red", 'X': "yellow-red", 'Y': "blue", 'Z': "magenta"}
                elif config.colour == "rgb":
                    dict_of_colour_with_character = {'1': "grey", '2': "grey", '3': "grey", '4': "grey", '5': "cyan", '6': "cyan", '7': "grey", '8': "grey", '9': "green", 
                    'A': "cyan", 'B': "brown", 'C': "orange", 'D': "blue", 'E': "blue", 'F': "blue", 'G': "yellow", 'H': "yellow", 'I': "green",'J': "green", 
                    'K': "green", 'L': "magenta", 'M': "magenta", 'N': "magenta", 'O': "green", 'P': "green", 'Q': "red", 'R': "blue", 'S': "cyan", 
                    'T': "blue", 'U': "green", 'V': "brown", 'W': "red", 'X': "orange", 'Y': "blue", 'Z': "magenta"}
                else:
                    print("please choose hsv or rgb")
                    break

                list_of_character = list(dict_of_colour_with_character.keys())
                list_of_colour = list(dict_of_colour_with_character.values())

                actual_colour = list_of_colour[list_of_character.index(str(actual_character))]

                test_image = cv2.imread(str(filename))

                t = time.time()
                color, roi, frame, success = detection(test_image, config)

                if success:
                    t0 += time.time() - t
                    t = time.time()
                    predicted_character, contour_image, chosen_image = character_recognition.character(color)
                    t1 += time.time() - t
                    t = time.time()
                    predicted_color, processed_image = colour_recognition.colour(color)
                    t2 += time.time() - t

                    seen += 1

                    if predicted_character == actual_character:
                        correct_prediction_of_character += 1
                    if predicted_color == actual_colour:
                        correct_prediction_of_colour += 1
                    
                    results_of_static_test(name_of_image, predicted_character, predicted_color, actual_character, actual_colour, save.save_dir, config)
                    
                    if config.save_results:
                        name_of_results = ["color", "roi", "frame","contour_image","processed_image", "chosen_image"]
                        image_results = [color, roi, frame, contour_image, processed_image, chosen_image]
                        for value, data in enumerate(name_of_results):
                            image_name = f"{name_of_image}_{data}.jpg"
                            image = image_results[value]
                            if image is not None:
                                save.save_the_image(image_name, image)

            percentage_of_correct_character = (correct_prediction_of_character/len(cap)) * 100
            percentage_of_correct_colour = (correct_prediction_of_colour/len(cap)) * 100

            if seen != 0: 
                speed = tuple(x / seen * 1E3 for x in (t0, t1, t2, t0 + t1 + t2))
            else:
                speed = [0.0, 0.0, 0.0, 0.0]

            with open(f'{save.save_dir}/{config.name_of_folder}_results_overall.csv', 'a') as csvfile:  # for testing purposes
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                filewriter.writerow([str(Path(config.media).stem), str(correct_prediction_of_character), str(percentage_of_correct_character), str(correct_prediction_of_colour), str(percentage_of_correct_colour), str(speed[0]), str(speed[1]), str(speed[2]), str(speed[3])])

            print(f"percentage_of_correct_character = {percentage_of_correct_character}")
            print(f"percentage_of_correct_colour = {percentage_of_correct_colour}")

            with open(f'{save.save_dir}/{config.name_of_folder}_information.txt', 'a') as f:
                f.write("Colour and Character Recognition for a 2 Square Target \n")
                f.write(f"Resolution of the images = 1080p \n")
                f.write(f"method used for character recognition = {config.character} \n")
                f.write(f"method used for colour recognition = {config.colour} \n")
                f.write("\n")


def main():
    print('Starting detection')
    capture_setting()


if __name__ == "__main__":
    main()
