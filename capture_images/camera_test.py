import cv2 
import time
import argparse

# check the ports you have ls /dev/video*
# V4L2 or khadas camera currently works for version 4.2.0 of opencv
# https://forum.khadas.com/t/view-simply-camera-opencv4-5/10741/4
# if you have other version of opencv then downgrade it

# remember to disable the IR sensor as it is enabled

def record_video(width, height, fps):
    val = True
    count = 0
    
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(cv2.CAP_V4L2)
    if  cap.isOpened()== False:
        print("camera port is inactive")
    else:
        print("camera port is active")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    time.sleep(2)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter("./test.avi", fourcc, fps, (width, height), True) # True means you want to record in colour https://pyimagesearch.com/2016/02/22/writing-to-video-with-opencv/
    
    while val is True:
        start = time.time()
        ret, frame = cap.read()
        end = time.time() - start
        cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        if frame is None:
            break
        else:
            # out.write(frame)
            print(f"{1/end}")
            # print("frame captured")
            # cv2.imshow("video", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27: 
                break
            if count < 300:
                count += 1
            if count == 300:
                # val = False
                print("hi")
    cap.release()
    out.release()
    
def capture_images(width, height, fps):
    
    val = True
    count = 0

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(cv2.CAP_V4L2)
    if  cap.isOpened()== False:
        print("camera port is inactive")
    else:
        print("camera port is active")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    while val is True:
        input("Are you Ready?")
        ret, frame = cap.read()
        cv2.imwrite(f"{count}.png", frame)
        cv2.imshow("distorted", frame)
        cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='camera_test.py')
    parser.add_argument('--video', action='store_true')
    opt = parser.parse_args()
    width = 1920
    height = 1080
    fps = 60
    
    if opt.video:
        record_video(width, height, fps)
    else:
        capture_images(width, height, fps)
