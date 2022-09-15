import picamera
from time import sleep
from subprocess import call
import argparse
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='pi_video.py')
    parser.add_argument('--name', type=str, default="", help='are you doing a distance test')
    opt = parser.parse_args()
    
    # Setup the camera
    with picamera.PiCamera() as camera:
        camera.framerate = 30
        camera.iso = 100 # https://expertphotography.com/indoor-photography-tips/
        sleep(2)
        camera.shutter_speed = camera.exposure_speed
        # Start recording
        print("Recording Started...")
        camera.start_recording(f"{opt.name}.h264")
        try:
            while True:
                sleep(0.1)
        except KeyboardInterrupt:
            # sleep(600)
            # Stop recording
            camera.stop_recording()
    
            # The camera is now closed.
            
            print("We are going to convert the video.")
            # Define the command we want to execute.
            command = f"MP4Box -add {opt.name}.h264 {opt.name}.mp4"
            # Execute our command
            call([command], shell=True)
            # Video converted.
    