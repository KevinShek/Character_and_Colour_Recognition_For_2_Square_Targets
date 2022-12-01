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
import datetime
from fractions import Fraction
from detection_methods import Detection

"""
The following code contains the detection of the square target and saves only the inner square data
"""

class MyClass():
    def __init__(self, param):
        self.param = param


def solution(counter, marker, predicted_character, predicted_color, result_dir):
    with open(f'{result_dir}/results.csv', 'a') as csvfile:  # for testing purposes
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        filewriter.writerow([str(marker), str(predicted_character), str(predicted_color)])

    print(f"detection of marker {marker} located")
    print(f"{predicted_character} is located for marker {marker}")
    print(f"{predicted_color} is the colour of ground marker {marker}")

    counter = 1
    config = Settings()
    if config.capture != "image":
        marker += 1

    return marker, counter


def results_of_real_detection_test(frame_name, predicted_character, actual_character, predicted_color, actual_color, result_dir):
    if predicted_character == actual_character:
        correct_character = 1
    else:
        correct_character = 0
    if predicted_color == actual_color:
        correct_colour = 1
    else:
        correct_colour = 0
        
    
    with open(f'{result_dir}/detection_results.csv', 'a') as csvfile:  # for testing purposes
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        filewriter.writerow([str(frame_name), str(predicted_character), str(actual_character), str(correct_character), str(predicted_color), str(actual_color), str(correct_colour)])
    
        

def results_of_distance_test(filename, predicted_character, predicted_color, actual_character, actual_color, result_dir, resolution_used, internal_folder):
    if predicted_character == actual_character:
        correct_character = 1
    else:
        correct_character = 0
    if predicted_color == actual_color:
        correct_colour = 1
    else:
        correct_colour = 0

    print(f"Predicted character and colour is: {predicted_character} and {predicted_color}")
    print(f"Actual character and colour is: {actual_character} and {actual_color}")

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

    print(f"Predicted character and colour is: {predicted_character} and {predicted_color}")
    print(f"Actual character and colour is: {actual_character} and {actual_color}")
    
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
    save = Saving(config.name_of_folder, config.exist_ok, are_you_testing = True)
    detect = Detection(config)


    
    # if config.detection_method == "vim3pro_method":
    #     from yolov4_detection import loading_model, detection
    #     loading_model(config)
    # else:
    #     from shape_detection import detection
        

    if config.capture == "camera":
        if config.testing == "video":
            cap = cv2.VideoCapture(config.source)
        else:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)  # 800 default
            cap.set(3, config.width)  # 800 default
            cap.set(4, config.height)  # 800 default
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
            cap.set(cv2.CAP_PROP_FPS, 60)
            time.sleep(2)  # allows the camera to start-up
            
        end_time_for_detection, end_time_for_character, end_time_for_colour = 0., 0., 0.
        seen = 0
        print('Camera on')
        if config.ready_check:
            distance = input("Are you Ready?")
        while True:
            if counter == 1 or end - start < 10:
                end = time.time()
                ret, frame = cap.read()
                if config.Step_camera:
                    cv2.imshow('frame', frame)
                    k = cv2.waitKey(5) & 0xFF
                    if k == 27:
                        break

                start_time_for_detection = time.time()
                # color, roi, frame, outer_edge, before_inner_edge_search, inner_edge, possible_target, success = detection(frame, config)
                detect.next_frame(frame)
                storing_inner_boxes_data = detect.storing_inner_boxes_data
                # storing_inner_boxes_data = detection(frame, config)
                end_time_for_detection = time.time() - start_time_for_detection
                seen_per_image_boolean = 0
                
                for i in range(int(len(storing_inner_boxes_data)/8)):                    
                    if storing_inner_boxes_data[7+(8*i)]:
                        if seen_per_image_boolean == 0:       
                            seen += 1
                            seen_per_image_boolean = 1
    
                        # time that the target has been last seen
                        start = time.time()
                        
                        start_time_for_character = time.time()
                        predicted_character, contour_image, chosen_image = character_recognition.character(storing_inner_boxes_data[0+(8*i)])
                        end_time_for_character = time.time() - start_time_for_character
                        
                        start_time_for_colour = time.time()
                        predicted_color, processed_image = colour_recognition.colour(storing_inner_boxes_data[0+(8*i)])
                        end_time_for_colour = time.time() - start_time_for_colour
                        
                        if config.testing == "real_time":                     
                            if predicted_character == actual_character:
                                correct_prediction_of_character += 1
                            if predicted_color == actual_colour:
                                correct_prediction_of_colour += 1
    
                        predicted_character_list.append(predicted_character)
                        predicted_color_list.append(predicted_color)
    
                        if config.save_results:
                            name_of_results = ["color", "roi", "frame","contour_image","processed_image", "chosen_image", "outer_edge", "inner_edge", "possible_target", "before_inner_edge_search"]
                            image_results = [storing_inner_boxes_data[0+(8*i)], storing_inner_boxes_data[1+(8*i)], storing_inner_boxes_data[2+(8*i)], contour_image, processed_image, chosen_image, storing_inner_boxes_data[3+(8*i)], storing_inner_boxes_data[5+(8*i)], storing_inner_boxes_data[6+(8+i)], storing_inner_boxes_data[4+(8*i)]]
                            for value, data in enumerate(name_of_results):
                                image_name = f"{marker}_{data}_{counter}.jpg"
                                image = image_results[value]
                                if image is not None:
                                    save.save_the_image(image_name, image)
                        counter = counter + 1
                        print("Target Captured and saved to file")

                if counter == 8:
                    print("Starting Recognition Thread")
                    common_character = Counter(predicted_character_list).most_common(1)[0][0]
                    common_color = Counter(predicted_color_list).most_common(1)[0][0]
                    marker, counter = solution(counter, marker, common_character, common_color, save.save_dir)
                    if config.testing == "real_time":
                        name = f"{marker}_{counter}"
                        results_of_real_detection_test(name, predicted_character, actual_character, predicted_color, actual_colour, save.save_dir)
                        detection_speed = end_time_for_detection / counter * 1E3

                        if seen != 0: 
                            recognition_speed = [x / seen * 1E3 for x in (end_time_for_character, end_time_for_colour)]
                            total_speed = detection_speed + recognition_speed[0] + recognition_speed[1]
                            speed = [detection_speed, recognition_speed[0], recognition_speed[1], total_speed]
                            percentage_of_correct_character = (correct_prediction_of_character/seen) * 100
                            percentage_of_correct_colour = (correct_prediction_of_colour/seen) * 100
                        else:
                            speed = [detection_speed, 0.0, 0.0, detection_speed]
                            percentage_of_correct_character, percentage_of_correct_colour = 0, 0
                        date = datetime.datetime.now()
                            
                        with open(f'{save.save_dir}/detection_results_overall.csv', 'a') as csvfile:  # for testing purposes
                            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                            filewriter.writerow([str(config.distance), str(seen), str(correct_prediction_of_character), str(percentage_of_correct_character), str(correct_prediction_of_colour), str(percentage_of_correct_colour), str(speed[0]), str(speed[1]), str(speed[2]), str(speed[3]), str(config.character), str(config.colour), str(camera.shutter_speed), str(camera.iso), str(camera.digital_gain), str(camera.analog_gain), str(red_gain), str(blue_gain), str(date)])
                            
                            
                        
                    predicted_character_list = []
                    predicted_color_list = []
                    seen = 0

            else:
                print("Starting Recognition Thread")
                common_character = Counter(predicted_character_list).most_common(1)[0][0]
                common_color = Counter(predicted_color_list).most_common(1)[0][0]
                marker, counter = solution(counter, marker, common_character, common_color, save.save_dir)
                if config.testing == "real_time":
                    name = f"{marker}_{counter}"
                    results_of_real_detection_test(name, predicted_character, actual_character, predicted_color, actual_colour, save.save_dir)
                    detection_speed = end_time_for_detection / counter * 1E3

                    if seen != 0: 
                        recognition_speed = [x / seen * 1E3 for x in (end_time_for_character, end_time_for_colour)]
                        total_speed = detection_speed + recognition_speed[0] + recognition_speed[1]
                        speed = [detection_speed, recognition_speed[0], recognition_speed[1], total_speed]
                        percentage_of_correct_character = (correct_prediction_of_character/seen) * 100
                        percentage_of_correct_colour = (correct_prediction_of_colour/seen) * 100
                    else:
                        speed = [detection_speed, 0.0, 0.0, detection_speed]
                        percentage_of_correct_character, percentage_of_correct_colour = 0, 0
                    date = datetime.datetime.now()
                        
                    with open(f'{save.save_dir}/detection_results_overall.csv', 'a') as csvfile:  # for testing purposes
                        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                        filewriter.writerow([str(config.distance), str(seen), str(correct_prediction_of_character), str(percentage_of_correct_character), str(correct_prediction_of_colour), str(percentage_of_correct_colour), str(speed[0]), str(speed[1]), str(speed[2]), str(speed[3]), str(config.character), str(config.colour), str(camera.shutter_speed), str(camera.iso), str(camera.digital_gain), str(camera.analog_gain), str(red_gain), str(blue_gain), str(date)])
                            
                    
                predicted_character_list = []
                predicted_color_list = []
                seen = 0
            # clear the stream in preparation for the next frame
            cap.truncate(0)

    elif config.capture == "pi_camera":
        from picamera.array import PiRGBArray
        from picamera import PiCamera
        from capture_images.set_picamera_gain import set_analog_gain, set_digital_gain

        camera = PiCamera()
        camera.resolution = (config.width, config.height)
        # camera.brightness = 50  # 50 is default
        camera.framerate = config.framerate
        camera.iso = config.iso 
        # camera.awb_mode = 'auto'
        time.sleep(2)
        camera.exposure_mode = 'off'
        g = camera.awb_gains
        camera.awb_mode = 'off'
        camera.awb_gains = g
        config.shutter_speed = camera.exposure_speed
        camera.shutter_speed = config.shutter_speed
        red_gain, blue_gain = camera.awb_gains
        end_time_for_detection, end_time_for_character, end_time_for_colour = 0., 0., 0.
        seen = 0
        
        if config.testing == "real_time":
            camera.awb_gains = config.red_gain, config.blue_gain # rounds to the nearest 1/256th  https://github.com/waveform80/picamera/issues/318
            set_digital_gain(camera, config.digital_gain)
            set_analog_gain(camera, config.analog_gain)
            red_gain, blue_gain = camera.awb_gains
            time.sleep(1)
            correct_prediction_of_character, correct_prediction_of_colour= 0, 0
            with open(f'{save.save_dir}/detection_results.csv', 'a') as csvfile:  # making the csv file
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                filewriter.writerow(["Filename", "Predicted Character", "Actual Character", "Correct Character", "Predicted Colour", "Actual Colour", "Correct Colour"])
            with open(f'{save.save_dir}/detection_results_overall.csv', 'a') as csvfile:  # for testing purposes
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                filewriter.writerow(["Distance","Detected Targets", "Correct Guess Character", "Correct Guess Character (%)", "Correct Guess Colour", "Correct Guess Colour (%)", "Detection Speed (ms)", "Character Recognition Speed (ms)", "Colour Recognition Speed (ms)", "Total Speed (ms)", f"Character Regnition Method", f"Colour Regnition Method", "Shutter Speed", "ISO", "Digial Gain", "Analog Gain", "Red Gain", "Blue Gain", "Timestamp"])
            actual_character = config.real_time_character
            actual_colour = config.real_time_colour
                


        cap = PiRGBArray(camera, size=(config.width, config.height))

        print("Camera Ready!")
        if config.ready_check:
            distance = input("Are you Ready?")
        
        try:
            for image in camera.capture_continuous(cap, format="bgr", use_video_port=True):
                #  to start the progress of capture and don't stop unless the counter increases and has surpass 5 seconds
                if counter == 1 or end - start < 10:
                    frame = image.array
                    end = time.time()
                    
                    start_time_for_detection = time.time()
    
                    # color, roi, frame, outer_edge, before_inner_edge_search, inner_edge, possible_target, success = detection(frame, config)
                    detect.next_frame(frame)
                    storing_inner_boxes_data = detect.storing_inner_boxes_data
                    # storing_inner_boxes_data = detection(frame, config)
                    end_time_for_detection = time.time() - start_time_for_detection
                    seen_per_image_boolean = 0
                    
                    for i in range(int(len(storing_inner_boxes_data)/8)):                    
                        if storing_inner_boxes_data[7+(8*i)]:
                            if seen_per_image_boolean == 0:       
                                seen += 1
                                seen_per_image_boolean = 1
        
                            # time that the target has been last seen
                            start = time.time()
                            
                            start_time_for_character = time.time()
                            predicted_character, contour_image, chosen_image = character_recognition.character(storing_inner_boxes_data[0+(8*i)])
                            end_time_for_character = time.time() - start_time_for_character
                            
                            start_time_for_colour = time.time()
                            predicted_color, processed_image = colour_recognition.colour(storing_inner_boxes_data[0+(8*i)])
                            end_time_for_colour = time.time() - start_time_for_colour
                            
                            if config.testing == "real_time":                     
                                if predicted_character == actual_character:
                                    correct_prediction_of_character += 1
                                if predicted_color == actual_colour:
                                    correct_prediction_of_colour += 1
        
                            predicted_character_list.append(predicted_character)
                            predicted_color_list.append(predicted_color)
        
                            if config.save_results:
                                name_of_results = ["color", "roi", "frame","contour_image","processed_image", "chosen_image", "outer_edge", "inner_edge", "possible_target", "before_inner_edge_search"]
                                image_results = [storing_inner_boxes_data[0+(8*i)], storing_inner_boxes_data[1+(8*i)], storing_inner_boxes_data[2+(8*i)], contour_image, processed_image, chosen_image, storing_inner_boxes_data[3+(8*i)], storing_inner_boxes_data[5+(8*i)], storing_inner_boxes_data[6+(8+i)], storing_inner_boxes_data[4+(8*i)]]
                                for value, data in enumerate(name_of_results):
                                    image_name = f"{marker}_{data}_{counter}.jpg"
                                    image = image_results[value]
                                    if image is not None:
                                        save.save_the_image(image_name, image)
                            counter = counter + 1
                            print("Target Captured and saved to file")
    
                    if counter == 8:
                        print("Starting Recognition Thread")
                        common_character = Counter(predicted_character_list).most_common(1)[0][0]
                        common_color = Counter(predicted_color_list).most_common(1)[0][0]
                        marker, counter = solution(counter, marker, common_character, common_color, save.save_dir)
                        if config.testing == "real_time":
                            name = f"{marker}_{counter}"
                            results_of_real_detection_test(name, predicted_character, actual_character, predicted_color, actual_colour, save.save_dir)
                            detection_speed = end_time_for_detection / counter * 1E3

                            if seen != 0: 
                                recognition_speed = [x / seen * 1E3 for x in (end_time_for_character, end_time_for_colour)]
                                total_speed = detection_speed + recognition_speed[0] + recognition_speed[1]
                                speed = [detection_speed, recognition_speed[0], recognition_speed[1], total_speed]
                                percentage_of_correct_character = (correct_prediction_of_character/seen) * 100
                                percentage_of_correct_colour = (correct_prediction_of_colour/seen) * 100
                            else:
                                speed = [detection_speed, 0.0, 0.0, detection_speed]
                                percentage_of_correct_character, percentage_of_correct_colour = 0, 0
                            date = datetime.datetime.now()
                                
                            with open(f'{save.save_dir}/detection_results_overall.csv', 'a') as csvfile:  # for testing purposes
                                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                                filewriter.writerow([str(config.distance), str(seen), str(correct_prediction_of_character), str(percentage_of_correct_character), str(correct_prediction_of_colour), str(percentage_of_correct_colour), str(speed[0]), str(speed[1]), str(speed[2]), str(speed[3]), str(config.character), str(config.colour), str(camera.shutter_speed), str(camera.iso), str(camera.digital_gain), str(camera.analog_gain), str(red_gain), str(blue_gain), str(date)])
                                
                                
                            
                        predicted_character_list = []
                        predicted_color_list = []
                        seen = 0
    
                else:
                    print("Starting Recognition Thread")
                    common_character = Counter(predicted_character_list).most_common(1)[0][0]
                    common_color = Counter(predicted_color_list).most_common(1)[0][0]
                    marker, counter = solution(counter, marker, common_character, common_color, save.save_dir)
                    if config.testing == "real_time":
                        name = f"{marker}_{counter}"
                        results_of_real_detection_test(name, predicted_character, actual_character, predicted_color, actual_colour, save.save_dir)
                        detection_speed = end_time_for_detection / counter * 1E3

                        if seen != 0: 
                            recognition_speed = [x / seen * 1E3 for x in (end_time_for_character, end_time_for_colour)]
                            total_speed = detection_speed + recognition_speed[0] + recognition_speed[1]
                            speed = [detection_speed, recognition_speed[0], recognition_speed[1], total_speed]
                            percentage_of_correct_character = (correct_prediction_of_character/seen) * 100
                            percentage_of_correct_colour = (correct_prediction_of_colour/seen) * 100
                        else:
                            speed = [detection_speed, 0.0, 0.0, detection_speed]
                            percentage_of_correct_character, percentage_of_correct_colour = 0, 0
                        date = datetime.datetime.now()
                            
                        with open(f'{save.save_dir}/detection_results_overall.csv', 'a') as csvfile:  # for testing purposes
                            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                            filewriter.writerow([str(config.distance), str(seen), str(correct_prediction_of_character), str(percentage_of_correct_character), str(correct_prediction_of_colour), str(percentage_of_correct_colour), str(speed[0]), str(speed[1]), str(speed[2]), str(speed[3]), str(config.character), str(config.colour), str(camera.shutter_speed), str(camera.iso), str(camera.digital_gain), str(camera.analog_gain), str(red_gain), str(blue_gain), str(date)])
                                
                        
                    predicted_character_list = []
                    predicted_color_list = []
                    seen = 0
                # clear the stream in preparation for the next frame
                cap.truncate(0)
                
        except KeyboardInterrupt:
            cap.truncate(0)
            camera.close()
            if seen != 0:
                common_character = Counter(predicted_character_list).most_common(1)[0][0]
                common_color = Counter(predicted_color_list).most_common(1)[0][0]
            else:
                common_character = None
                common_color = None
                predicted_character = None
                predicted_color = None
                
            solution(counter, marker, common_character, common_color, save.save_dir)
            if config.testing == "real_time":
                name = f"{marker}_{counter}"
                results_of_real_detection_test(name, predicted_character, actual_character, predicted_color, actual_colour, save.save_dir)

                detection_speed = end_time_for_detection / counter * 1E3

                if seen != 0: 
                    recognition_speed = [x / seen * 1E3 for x in (end_time_for_character, end_time_for_colour)]
                    total_speed = detection_speed + recognition_speed[0] + recognition_speed[1]
                    speed = [detection_speed, recognition_speed[0], recognition_speed[1], total_speed]
                    percentage_of_correct_character = (correct_prediction_of_character/seen) * 100
                    percentage_of_correct_colour = (correct_prediction_of_colour/seen) * 100
                else:
                    speed = [detection_speed, 0.0, 0.0, detection_speed]
                    percentage_of_correct_character, percentage_of_correct_colour = 0, 0
                    
                date = datetime.datetime.now()
                    
                with open(f'{save.save_dir}/detection_results_overall.csv', 'a') as csvfile:  # for testing purposes
                    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                    filewriter.writerow([str(config.distance), str(seen), str(correct_prediction_of_character), str(percentage_of_correct_character), str(correct_prediction_of_colour), str(percentage_of_correct_colour), str(speed[0]), str(speed[1]), str(speed[2]), str(speed[3]), str(config.character), str(config.colour), str(config.shutter_speed), str(config.iso), str(config.digital_gain), str(config.analog_gain), str(red_gain), str(blue_gain), str(date)])
            
            print("Progress saved and exiting the program.")
                                
            
                    
    elif config.capture == "image":
        if config.testing == " ":
            cap = [] # to store the names of the images
            data_dir = Path(config.source)

            # the following code interite over the extension that exist within a folder and place them into a single list
            image_count = list(itertools.chain.from_iterable(data_dir.glob(pattern) for pattern in ('*.jpg', '*.png')))
            # image_count = len(list(data_dir.glob('*.jpg')))
            for name in image_count:
                    # head, tail = ntpath.split(name)
                    filename = Path(name)  # .stem removes the extension and .name grabs the filename with extension
                    cap.append(filename)
                    test_image = cv2.imread(str(filename))
                    marker = Path(name).stem # grabs the name with the extension

                    # color, roi, frame, outer_edge, before_inner_edge_search, inner_edge, possible_target, success = detection(test_image, config)
                    detect.next_frame(test_image)
                    storing_inner_boxes_data = detect.storing_inner_boxes_data
                    # storing_inner_boxes_data = detection(test_image, config)
                    
                    for i in range(int(len(storing_inner_boxes_data)/8)):                    
                        if storing_inner_boxes_data[7+(8*i)]:
                            predicted_character, contour_image, chosen_image = character_recognition.character(storing_inner_boxes_data[0+(8*i)])
                            predicted_color, processed_image = colour_recognition.colour(storing_inner_boxes_data[0+(8*i)])

                            _, _ = solution(counter, marker, predicted_character, predicted_color, save.save_dir)
        
                        else:
                            contour_image = None
                            processed_image = None
                            chosen_image = None

                        if config.save_results:
                            name_of_results = ["color", "roi", "frame","contour_image","processed_image", "chosen_image", "outer_edge", "inner_edge", "possible_target", "before_inner_edge_search"]
                            image_results = [storing_inner_boxes_data[0+(8*i)], storing_inner_boxes_data[1+(8*i)], storing_inner_boxes_data[2+(8*i)], contour_image, processed_image, chosen_image, storing_inner_boxes_data[3+(8*i)], storing_inner_boxes_data[5+(8*i)], storing_inner_boxes_data[6+(8*i)], storing_inner_boxes_data[4+(8*i)]]
                            for value, data in enumerate(name_of_results):
                                image_name = f"{marker}_{data}_{i}.jpg"
                                image = image_results[value]
                                if image is not None:
                                    save.save_the_image(image_name, image)

                        print("Detected and saved a target")
            print(f"there is a total image count of {len(image_count)} and frames appended {len(cap)}")

        if config.testing == "distance_test":
            data_dir = Path(config.source)
            resolution_used = f"{config.source}".rsplit('/', 1)[-1] # grabbing the name of the folder name

            with open(f'{save.save_dir}/{resolution_used}_results.csv', 'a') as csvfile:  # making the csv file
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                filewriter.writerow(["Filename", "Predicted Character", "Actual Character", "Correct Character", "Predicted Colour", "Actual Colour", "Correct Colour"])
            with open(f'{save.save_dir}/{resolution_used}_results_overall.csv', 'a') as csvfile:  # for testing purposes
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                filewriter.writerow(["Distance","Detected Targets", "Correct Guess Character", "Correct Guess Character (%)", "Correct Guess Colour", "Correct Guess Colour (%)", "Detection Speed (ms)", "Character Recognition Speed (ms)", "Colour Recognition Speed (ms)", "Total Speed (ms)", f"Character Regnition Method ({config.character})", f"Colour Regnition Method ({config.colour})"])

            list_of_internal_folders = sorted(os.listdir(data_dir)) # sorted into numerical order

            for internal_folder in list_of_internal_folders:
                cap = [] # to store the names of the images
                correct_prediction_of_character, correct_prediction_of_colour= 0, 0
                t0, t1, t2 = 0., 0., 0.
                seen = 0
                print(f"distance = {internal_folder}m")

                # making internal folders for each distances
                Path(f"{save.save_dir}/{internal_folder}").mkdir(parents=True, exist_ok=True)

                data_dir = Path(f"{config.source}/{internal_folder}") # path to the images where it is located
                # the following code interite over the extension that exist within a folder and place them into a single list
                image_count = sorted(list(itertools.chain.from_iterable(data_dir.glob(pattern) for pattern in ('*.jpg', '*.png'))))

                # image_count = len(list(data_dir.glob('*.jpg')))
                for name in image_count:
                    # head, tail = ntpath.split(name)
                    filename = Path(name)  # .stem removes the extension and .name grabs the filename with extension
                    name_of_image = filename.stem
                    # print(name_of_image)
                    actual_character = f"{name_of_image}".rsplit('_', 5)[0]
                    option = f"{name_of_image}".rsplit('_', 5)[-1]

                    if option != config.testing_set:
                        continue # skips the current iteration

                    cap.append(filename) # append the path of the images to the list

                    if config.colour == "hsv":
                        dict_of_colour_with_character = {'1': "red", '2': "green", '3': "blue", '4': "red", '5': "green", '6': "blue", '7': "red", '8': "green", '9': "blue", 
                        'A': "red", 'B': "green", 'C': "blue", 'D': "red", 'E': "green", 'F': "blue", 'G': "red", 'H': "green", 'I': "blue",'J': "red", 
                        'K': "green", 'L': "blue", 'M': "red", 'N': "green", 'O': "blue", 'P': "red", 'Q': "green", 'R': "blue", 'S': "red", 
                        'T': "green", 'U': "blue", 'V': "red", 'W': "green", 'X': "blue", 'Y': "red", 'Z': "green"}
                    elif config.colour == "rgb":
                        dict_of_colour_with_character = {'1': "red", '2': "green", '3': "blue", '4': "red", '5': "green", '6': "blue", '7': "red", '8': "green", '9': "blue", 
                        'A': "red", 'B': "green", 'C': "blue", 'D': "red", 'E': "green", 'F': "blue", 'G': "red", 'H': "green", 'I': "blue",'J': "red", 
                        'K': "green", 'L': "blue", 'M': "red", 'N': "green", 'O': "blue", 'P': "red", 'Q': "green", 'R': "blue", 'S': "red", 
                        'T': "green", 'U': "blue", 'V': "red", 'W': "green", 'X': "blue", 'Y': "red", 'Z': "green"}
                    else:
                        print("please choose hsv or rgb")
                        break

                    list_of_character = list(dict_of_colour_with_character.keys())
                    list_of_colour = list(dict_of_colour_with_character.values())

                    actual_colour = list_of_colour[list_of_character.index(str(actual_character))]

                    test_image = cv2.imread(str(filename))

                    t = time.time()
                    # color, roi, frame, outer_edge, before_inner_edge_search, inner_edge, possible_target, success = detection(test_image, config)
                    detect.next_frame(test_image)
                    storing_inner_boxes_data = detect.storing_inner_boxes_data
                    # storing_inner_boxes_data = detection(test_image, config)
                    print(f"number of boxes detected = {len(storing_inner_boxes_data)/8}")
                    # print(storing_inner_boxes_data)      
                    t0 += time.time() - t
                    seen_per_image_boolean = 0
                    end_time_character, end_time_colour = [], []

                    for i in range(int(len(storing_inner_boxes_data)/8)):              
                        if storing_inner_boxes_data[7+(8*i)]:
                            if seen_per_image_boolean == 0:
                                seen += 1
                                seen_per_image_boolean = 1
                                
                            t = time.time()
                            predicted_character, contour_image, chosen_image = character_recognition.character(storing_inner_boxes_data[0+(8*i)])
                            predicted_character = predicted_character.capitalize()
                            end_time_character.append(time.time() - t)
                            # t1 += time.time() - t
                            t = time.time()
                            predicted_color, processed_image = colour_recognition.colour(storing_inner_boxes_data[0+(8*i)])
                            end_time_colour.append(time.time() - t)
                            # t2 += time.time() - t

                            if predicted_character == actual_character:
                                correct_prediction_of_character += 1
                            if predicted_color == actual_colour:
                                correct_prediction_of_colour += 1
                            
                            results_of_distance_test(name_of_image, predicted_character, predicted_color, actual_character, actual_colour, save.save_dir, resolution_used, internal_folder)
                        
                        else:
                            contour_image = None
                            processed_image = None
                            chosen_image = None
    
                        if config.save_results:
                            name_of_results = ["color", "roi", "frame","contour_image","processed_image", "chosen_image", "outer_edge", "inner_edge", "possible_target", "before_inner_edge_search"]
                            image_results = [storing_inner_boxes_data[0+(8*i)], storing_inner_boxes_data[1+(8*i)], storing_inner_boxes_data[2+(8*i)], contour_image, processed_image, chosen_image, storing_inner_boxes_data[3+(8*i)], storing_inner_boxes_data[5+(8*i)], storing_inner_boxes_data[6+(8*i)], storing_inner_boxes_data[4+(8*i)]]
                            for value, data in enumerate(name_of_results):
                                image_name = f"{internal_folder}/{name_of_image}_{data}_{i}.jpg"
                                image = image_results[value]
                                if image is not None:
                                    save.save_the_image(image_name, image)
                                    
                    if seen_per_image_boolean == 1:
                        average_time_character = sum(end_time_character)/len(end_time_character)
                        average_time_colour = sum(end_time_colour)/len(end_time_colour)
                        t1 += average_time_character
                        t2 += average_time_colour

                percentage_of_correct_character = (correct_prediction_of_character/len(cap)) * 100
                percentage_of_correct_colour = (correct_prediction_of_colour/len(cap)) * 100

                detection_speed = t0 / len(cap) * 1E3

                if seen != 0: 
                    recognition_speed = [x / seen * 1E3 for x in (t1, t2)]
                    total_speed = detection_speed + recognition_speed[0] + recognition_speed[1]
                    speed = [detection_speed, recognition_speed[0], recognition_speed[1], total_speed]
                else:
                    speed = [detection_speed, 0.0, 0.0, detection_speed]

                with open(f'{save.save_dir}/{resolution_used}_results_overall.csv', 'a') as csvfile:  # for testing purposes
                    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                    filewriter.writerow([str(internal_folder), str(seen), str(correct_prediction_of_character), str(percentage_of_correct_character), str(correct_prediction_of_colour), str(percentage_of_correct_colour), str(speed[0]), str(speed[1]), str(speed[2]), str(speed[3])])

                print(f"percentage_of_correct_character = {percentage_of_correct_character}")
                print(f"percentage_of_correct_colour = {percentage_of_correct_colour}")
                print(f"detection speed = {detection_speed}")

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
            data_dir = Path(config.source)

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
                name_of_image = filename.stem
                # print(name_of_image)
                actual_character = f"{name_of_image}".rsplit('_', 5)[0]
                option = f"{name_of_image}".rsplit('_', 5)[-1]

                if option != config.testing_set:
                    continue # skips the current iteration

                cap.append(filename) # append the path of the images to the list

                if config.colour == "hsv":
                    dict_of_colour_with_character = {'1': "red", '2': "green", '3': "blue", '4': "red", '5': "green", '6': "blue", '7': "red", '8': "green", '9': "blue", 
                        'A': "red", 'B': "green", 'C': "blue", 'D': "red", 'E': "green", 'F': "blue", 'G': "red", 'H': "green", 'I': "blue",'J': "red", 
                        'K': "green", 'L': "blue", 'M': "red", 'N': "green", 'O': "blue", 'P': "red", 'Q': "green", 'R': "blue", 'S': "red", 
                        'T': "green", 'U': "blue", 'V': "red", 'W': "green", 'X': "blue", 'Y': "red", 'Z': "green"}
                elif config.colour == "rgb":
                    dict_of_colour_with_character = {'1': "red", '2': "green", '3': "blue", '4': "red", '5': "green", '6': "blue", '7': "red", '8': "green", '9': "blue", 
                        'A': "red", 'B': "green", 'C': "blue", 'D': "red", 'E': "green", 'F': "blue", 'G': "red", 'H': "green", 'I': "blue",'J': "red", 
                        'K': "green", 'L': "blue", 'M': "red", 'N': "green", 'O': "blue", 'P': "red", 'Q': "green", 'R': "blue", 'S': "red", 
                        'T': "green", 'U': "blue", 'V': "red", 'W': "green", 'X': "blue", 'Y': "red", 'Z': "green"}
                else:
                    print("please choose hsv or rgb")
                    break

                list_of_character = list(dict_of_colour_with_character.keys())
                list_of_colour = list(dict_of_colour_with_character.values())

                actual_colour = list_of_colour[list_of_character.index(str(actual_character))]

                test_image = cv2.imread(str(filename))

                t = time.time()
                # color, roi, frame, outer_edge, before_inner_edge_search, inner_edge, possible_target, success = detection(test_image, config)
                detect.next_frame(test_image)
                storing_inner_boxes_data = detect.storing_inner_boxes_data
                storing_inner_boxes_data = detection(test_image, config)
                t0 += time.time() - t
                seen_per_image_boolean = 0

                for i in range(int(len(storing_inner_boxes_data)/8)):                    
                    if storing_inner_boxes_data[7+(8*i)]:
                        if seen_per_image_boolean == 0:
                            seen += 1
                            seen_per_image_boolean = 1
                            t = time.time()
                            predicted_character, contour_image, chosen_image = character_recognition.character(storing_inner_boxes_data[0+(8*i)])
                            predicted_character = predicted_character.capitalize()
                            t1 += time.time() - t
                            t = time.time()
                            predicted_color, processed_image = colour_recognition.colour(storing_inner_boxes_data[0+(8*i)])
                            t2 += time.time() - t
        
                            if predicted_character == actual_character:
                                correct_prediction_of_character += 1
                            if predicted_color == actual_colour:
                                correct_prediction_of_colour += 1
                            
                            results_of_static_test(filename, predicted_character, predicted_color, actual_character, actual_colour, save.save_dir, config)
                    
                    else:
                        contour_image = None
                        processed_image = None
                        chosen_image = None

                    if config.save_results:
                        name_of_results = ["color", "roi", "frame","contour_image","processed_image", "chosen_image", "outer_edge", "inner_edge", "possible_target", "before_inner_edge_search"]
                        image_results = [storing_inner_boxes_data[0+(8*i)], storing_inner_boxes_data[1+(8*i)], storing_inner_boxes_data[2+(8*i)], contour_image, processed_image, chosen_image, storing_inner_boxes_data[3+(8*i)], storing_inner_boxes_data[5+(8*i)], storing_inner_boxes_data[6+(8*i)], storing_inner_boxes_data[4+(8*i)]]
                        for value, data in enumerate(name_of_results):
                            image_name = f"{name_of_image}_{data}_{i}.jpg"
                            image = image_results[value]
                            if image is not None:
                                save.save_the_image(image_name, image)


            percentage_of_correct_character = (correct_prediction_of_character/len(cap)) * 100
            percentage_of_correct_colour = (correct_prediction_of_colour/len(cap)) * 100

            detection_speed = t0 / len(cap) * 1E3

            if seen != 0: 
                recognition_speed = [x / seen * 1E3 for x in (t1, t2)]
                total_speed = detection_speed + recognition_speed[0] + recognition_speed[1]
                speed = [detection_speed, recognition_speed[0], recognition_speed[1], total_speed]
            else:
                speed = [detection_speed, 0.0, 0.0, detection_speed]

            with open(f'{save.save_dir}/{config.name_of_folder}_results_overall.csv', 'a') as csvfile:  # for testing purposes
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                filewriter.writerow([str(Path(config.source).stem), str(correct_prediction_of_character), str(percentage_of_correct_character), str(correct_prediction_of_colour), str(percentage_of_correct_colour), str(speed[0]), str(speed[1]), str(speed[2]), str(speed[3])])

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
