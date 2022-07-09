from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import argparse
from io import BytesIO
import numpy as np
import csv
from pathlib import Path
import datetime
from fractions import Fraction
from set_picamera_gain import set_analog_gain, set_digital_gain


def pi_capture(test_name, resolution, distance, character, camera, path_of_folder): 

    print(f"starting capture of character: {character} for {resolution}")
    if resolution == "phone_resolution":
        camera.resolution = (2560, 1152)
    if resolution == "1080p":
        camera.resolution = (1920, 1080)
    elif resolution == "720p":
        camera.resolution = (1280, 720)
    elif resolution == "480p":
        camera.resolution = (640, 480)
    elif resolution == "8m":
    # to use this, you should enable gpu via the config: https://stackoverflow.com/questions/39251815/python-not-taking-picture-at-highest-resolution-from-raspberry-pi-camera
        camera.resolution = (3280, 2464)
#        camera.capture(f"/home/pi/test/benchmark/{resolution}/{distance}/{character}_{distance}_{resolution}.png")
        # return
        
        
    if test_name == "distance_test":
          camera.capture(f"/home/pi/test/benchmark/{resolution}/{distance}/{character}_{distance}_{resolution}_still_image.png")
    elif test_name == "static_test":
          camera.capture(f"/home/pi/test/benchmark/static/{character}_{resolution}_still_image.png")
    else:
          camera.capture(f"/home/pi/test/benchmark/photos/{character}_{resolution}_still_image.png")
          
               
    cap = PiRGBArray(camera, size=(camera.resolution[0], camera.resolution[1]))
   
    for image in camera.capture_continuous(cap, format="bgr", use_video_port=True):
      frame = image.array
    
      if test_name == "distance_test":
          filepath = f"/home/pi/test/benchmark/{resolution}/{distance}/{character}_{distance}_{resolution}_video.png"
      elif test_name == "static_test":
          filepath = f"/home/pi/test/benchmark/static/{character}_{resolution}_video.png"
      else:
          filepath = f"/home/pi/test/benchmark/photos/{character}_{resolution}_video.png"
          
      cv2.imwrite(filepath, frame)
      
      break
      
    print(f"digatal gain and analog gain = {camera.digital_gain} and {camera.analog_gain}") # this is fixed due to exposure mode being switched off
    red_gain, blue_gain = camera.awb_gains
    date = datetime.datetime.now()
    
    print(f"awb gains = {camera.awb_gains}") # provides values in fractions e.g. Fraction(45, 32) = 45/32
    print(f"iso = {camera.iso}")
    print(f"shutter speed = {camera.shutter_speed}")
    print(f"Captured at {date}")
    
    with open(f'{path_of_folder}/results.csv', 'a') as csvfile:  # for testing purposes
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        filewriter.writerow([str(character), str(camera.shutter_speed), str(camera.iso), str(camera.digital_gain), str(camera.analog_gain), str(red_gain), str(blue_gain), str(date)])
    
    print("finish capture")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='pi_capture.py')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_name', type=str, default="", help='are you doing a distance test')
    parser.add_argument('--resolution', type=str, default='1080p', help='camera resolution')
    parser.add_argument('--distance', type=str, help='what is the distance')
    parser.add_argument('--character', type=str, help='what is the character')
    opt = parser.parse_args()
    
    # camera.awb_mode = "auto" # not a good selection as it will adjust accordingly to the environment
    # print(f"fluorescent = {camera.awb_gains}")
    # camera.awb_mode = "sunlight"
    # print(f"sunlight = {camera.awb_gains}")
    # camera.awb_gains = (4, 4) # https://picamera.readthedocs.io/en/release-1.13/api_camera.html#picamera.PiCamera.awb_gains

    # camera.exposure_mode = "auto"
    # camera.iso = 0
    # camera setting for fixing https://picamera.readthedocs.io/en/release-1.13/recipes1.html#capturing-consistent-images
    camera = PiCamera(framerate=30) # framerate 30 is default value
    camera.iso = 100 # https://expertphotography.com/indoor-photography-tips/
    # camera.awb_mode = "fluorescent"
    time.sleep(2)
    
    camera.shutter_speed = camera.exposure_speed
    # camera.shutter_speed = int(1E6/camera.framerate) # equvilent to 1/fps (https://picamera.readthedocs.io/en/release-1.13/api_camera.html#picamera.PiCamera.shutter_speed)
    camera.exposure_mode = 'off'
    g = camera.awb_gains
    camera.awb_mode = 'off'
    if opt.test:
      if opt.test_name == "distance_test":
        camera.awb_gains = Fraction(151, 107), Fraction(281, 128)
        set_digital_gain(camera, 1)
        set_analog_gain(camera, Fraction(331, 128))
    else:
        camera.awb_gains = g
    
    """# calbration of camera
    camera.resolution = (1280, 720)
    camera.awb_mode = 'off'
    # Start off with ridiculously low gains
    rg, bg = (0.5, 0.5)
    camera.awb_gains = (rg, bg)
    with PiRGBArray(camera, size=(128, 72)) as output:
      # Allow 30 attempts to fix AWB
      for i in range(30):
          # Capture a tiny resized image in RGB format, and extract the
          # average R, G, and B values
          camera.capture(output, format='rgb', resize=(128, 72), use_video_port=True)
          r, g, b = (np.mean(output.array[..., i]) for i in range(3))
          print('R:%5.2f, B:%5.2f = (%5.2f, %5.2f, %5.2f)' % (
              rg, bg, r, g, b))
          # Adjust R and B relative to G, but only if they're significantly
          # different (delta +/- 2)
          if abs(r - g) > 2:
              if r > g:
                  rg -= 0.1
              else:
                  rg += 0.1
          if abs(b - g) > 1:
              if b > g:
                  bg -= 0.1
              else:
                  bg += 0.1
          camera.awb_gains = (rg, bg)
          output.seek(0)
          output.truncate()
      
      print(rg, bg)"""
      
    if opt.test:
      if opt.character == None:
        list_of_characters = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I','J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 
                          'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
      else:
        list_of_characters = [opt.character]
      if opt.test_name == "distance_test":      
        list_of_exposure_settings = ["off"]
        list_of_resolutions = ["480p", "720p", "1080p", "8m"]
        for character in list_of_characters:
          input(f"Press Enter when you have the Character {character} ready!")
          for resolution in list_of_resolutions:
            for exposure_setting in list_of_exposure_settings:
              path_of_folder = Path(Path("benchmark") / resolution / opt.distance)
              camera.exposure_mode = exposure_setting
              if not character == "1" and resolution == "8m":
                continue
              if character == "1":
                with open(f'{path_of_folder}/results.csv', 'a') as csvfile:  # making the csv file
                  filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                  filewriter.writerow(["Character", "Shutter Speed", "ISO", "Digial Gain", "Analog Gain", "Red Gain", "Blue Gain", "Timestamp"])
                print(f"csv was made at {path_of_folder}")
              pi_capture(opt.test_name, resolution, opt.distance, character, camera, path_of_folder)
      elif opt.test_name == "static_test":
        list_of_exposure_settings = ["off"]
        for character in list_of_characters:
          input(f"Press Enter when you have the Character {character} ready!")
          for exposure_setting in list_of_exposure_settings:
            path_of_folder = Path(Path("benchmark") / "static")
            if character == "1":
                with open(f'{path_of_folder}/results.csv', 'a') as csvfile:  # making the csv file
                  filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                  filewriter.writerow(["Character", "Shutter Speed", "ISO", "Digial Gain", "Analog Gain", "Red Gain", "Blue Gain","Timestamp"])
            camera.exposure_mode = exposure_setting
            pi_capture(opt.test_name, opt.resolution, opt.distance, character, camera, path_of_folder)
    else:
      path_of_folder = Path(Path("benchmark") / "photos")
      pi_capture(opt.test_name, opt.resolution, opt.distance, opt.character, camera, path_of_folder)

