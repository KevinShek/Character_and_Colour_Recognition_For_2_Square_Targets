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

"""
The following code contains the detection of the square target and saves only the inner square data
"""


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
    
    
def capture_setting():
    # intialising the key information
    val = True
    counter = 1
    marker = 1
    distance = 0
    predicted_character_list = []
    predicted_color_list = []
    end = time.time()
    start = time.time()
    config = Settings()
    save = Saving(config.name_of_folder, config.exist_ok)
    
    if config.detection_method == "vim3pro_method":
        from yolov4_detection import loading_model, detection
        loading_model(config)
    else:
        from shape_detection import detection
        
    webcam = config.capture.isnumeric() or config.capture.endswith('.txt') or config.capture.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')) or config.capture.endswith('.mp4')
    
    if webcam:
        pipe = eval(config.capture)
        print(pipe)
        cap = cv2.VideoCapture(pipe)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)  # 800 default
        # cap.set(3, config.width)  # 800 default
        # cap.set(4, config.height)  # 800 default
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        time.sleep(2)  # allows the camera to start-up
        
        end_time_for_detection, end_time_for_character, end_time_for_colour = 0., 0., 0.
        seen = 0
        print('Camera on')
        if config.ready_check:
            distance = input("Are you Ready?")
        while val:
            if cv2.waitKey(33) == ord('a'):
                val = False                
            ret, frame = cap.read()
            cv2.imshow("frame", frame)
            start_time_for_detection = time.time()
            storing_inner_boxes_data = detection(frame, config)
            end_time_for_detection = time.time() - start_time_for_detection
            
            for i in range(int(len(storing_inner_boxes_data)/8)):     
                if config.recognition_method:                  
                    if storing_inner_boxes_data[7+(8*i)]:    
                          seen += 1
  
                          # time that the target has been last seen
                          start = time.time()
                          
                          start_time_for_character = time.time()
                          predicted_character, contour_image, chosen_image = character_recognition.character(storing_inner_boxes_data[0+(8*i)])
                          end_time_for_character = time.time() - start_time_for_character
                          
                          start_time_for_colour = time.time()
                          predicted_color, processed_image = colour_recognition.colour(storing_inner_boxes_data[0+(8*i)])
                          end_time_for_colour = time.time() - start_time_for_colour
      
                          predicted_character_list.append(predicted_character)
                          predicted_color_list.append(predicted_color)
                    else:
                        contour_image, chosen_image, processed_image, predicted_character, predicted_color = None, None, None, None, None
                else:
                    contour_image, chosen_image, processed_image, predicted_character, predicted_color = None, None, None, None, None
  
                if config.save_results:
                    name_of_results = ["color", "roi", "frame","contour_image","processed_image", "chosen_image", "outer_edge", "inner_edge", "possible_target", "before_inner_edge_search"]
                    image_results = [storing_inner_boxes_data[0+(8*i)], storing_inner_boxes_data[1+(8*i)], storing_inner_boxes_data[2+(8*i)], contour_image, processed_image, chosen_image, storing_inner_boxes_data[3+(8*i)], storing_inner_boxes_data[5+(8*i)], storing_inner_boxes_data[6+(8*i)], storing_inner_boxes_data[4+(8*i)]]
                    for value, data in enumerate(name_of_results):
                        image_name = f"{marker}_{data}_{counter}.jpg"
                        image = image_results[value]
                        if image is not None:
                            save.save_the_image(image_name, image)
                counter = counter + 1
                print("Target Captured and saved to file")
                fps = 1/end_time_for_detection
                with open(f'{save.save_dir}/results.csv', 'a') as csvfile:  # for testing purposes
                    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                    filewriter.writerow([str(counter), str(end_time_for_detection), str(fps), str(len(storing_inner_boxes_data)/8), str(predicted_character), str(predicted_color)])
                    
        # clear the stream in preparation for the next frame
        cap.release()
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
        
        cap = PiRGBArray(camera, size=(config.width, config.height))

        print("Camera Ready!")
        if config.ready_check:
            distance = input("Are you Ready?")
      
        try:
            for image in camera.capture_continuous(cap, format="bgr", use_video_port=True):
                if cv2.waitKey(33) == ord('a'):
                    break             
                frame = image.array
                start_time_for_detection = time.time()
                storing_inner_boxes_data = detection(frame, config)
                end_time_for_detection = time.time() - start_time_for_detection

                for i in range(int(len(storing_inner_boxes_data)/8)):     
                    if config.recognition_method:                  
                        if storing_inner_boxes_data[7+(8*i)]:    
                              seen += 1
      
                              # time that the target has been last seen
                              start = time.time()
                              
                              start_time_for_character = time.time()
                              predicted_character, contour_image, chosen_image = character_recognition.character(storing_inner_boxes_data[0+(8*i)])
                              end_time_for_character = time.time() - start_time_for_character
                              
                              start_time_for_colour = time.time()
                              predicted_color, processed_image = colour_recognition.colour(storing_inner_boxes_data[0+(8*i)])
                              end_time_for_colour = time.time() - start_time_for_colour
          
                              predicted_character_list.append(predicted_character)
                              predicted_color_list.append(predicted_color)
                        else:
                            contour_image, chosen_image, processed_image, predicted_character, predicted_color = None, None, None, None, None
                    else:
                        contour_image, chosen_image, processed_image, predicted_character, predicted_color = None, None, None, None, None
        
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
                    fps = 1/end_time_for_detection
                    with open(f'{save.save_dir}/results.csv', 'a') as csvfile:  # for testing purposes
                        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                        filewriter.writerow([str(counter), str(end_time_for_detection), str(fps), str(len(storing_inner_boxes_data)/8), str(predicted_character), str(predicted_color)])
    
                # clear the stream in preparation for the next frame
                cap.truncate(0)
        except:
            print("System Shutdown")


def main():
    print('Starting detection')
    capture_setting()


if __name__ == "__main__":
    main()
