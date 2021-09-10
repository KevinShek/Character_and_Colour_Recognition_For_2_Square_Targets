import cv2
import time
import os
from pathlib import Path
import glob
import re


class Saving:
    def __init__(self, testing, exist_ok=False):
        # self.save_dir = save_dir
        # Directories
        self.save_dir = Path(self.increment_path(Path("result_dir") / testing, exist_ok=exist_ok))  # increment run
        (self.save_dir / 'labels' if False else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir


    def increment_path(self, path, exist_ok=True, sep=''):
        # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
        path = Path(path)  # os-agnostic
        if (path.exists() and exist_ok) or (not path.exists()):
            return str(path)
        else:
            dirs = glob.glob(f"{path}{sep}*")  # similar paths
            matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
            i = [int(m.groups()[0]) for m in matches if m]  # indices
            n = max(i) + 1 if i else 2  # increment number
            return f"{path}{sep}{n}"  # update path


    # Has to be intialised before attempting for video saving, for mp4 use "MP4V" and for avi use "XVID"
    def name_of_output_for_video(self, testing, cap, is_video):
        # for video saving
        self.output_file = f"{testing}" + ".mp4"
        save_path = str(self.save_dir / self.output_file)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size = (width, height)
        if is_video:
            self.video = self.set_saved_video(cap, save_path, size)
        else:
            self.video = self.stream(size)


    def stream(self, size):
        # this requires the source of Opencv with gstreamer downloaded with it
        # sending data to a tcp protocol
        pipeline = "appsrc ! video/x-raw, format=BGR ! queue ! videoconvert ! x264enc noise-reduction=10 speed-preset=ultrafast tune=zerolatency ! h264parse ! rtph264pay config-interval=1 pt=96 ! udpsink host=127.0.0.1 port=8080 sync=true"
        framerate = 25
        video = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, framerate, size, True)
        return video


    def set_saved_video(self, input_video, output_video, size):
        # for mp4 use MP4V for avi use XVID
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(input_video.get(cv2.CAP_PROP_FPS))
        video = cv2.VideoWriter(output_video, fourcc, fps, size)
        return video


    def saving_the_frame(self, image_np):
        self.video.write(image_np)


    def save_the_image(self, image_name, image):
        """
        For writing an image file to the specified directory with standardised file name format
        - image_name - name of file
        - image - saving the image of interest
        """

        # Form file name by extracting the name by locating the last slash
        # file_name = f"{image_name}".rsplit('/', 1)[-1]

        # Form full path
        filepath = os.path.join(self.save_dir, image_name)
        # Write file
        cv2.imwrite(filepath, image)
