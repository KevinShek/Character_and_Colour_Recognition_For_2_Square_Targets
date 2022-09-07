import numpy as np
import cv2
from pathlib import Path
import itertools

## calbrating the distorted camera based on saved images for calbration 
def calbrate_distorted_camera_based_on_images(calbrate_distort_camera_path):
    term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    obj_points = np.zeros((4*5,3), np.float32)
    obj_points[:,:2] = np.mgrid[0:5,0:4].T.reshape(-1,2)
    real_points = []
    img_points = []
    chess_images_path = Path(calbrate_distort_camera_path)
    # print(chess_images_path)
    chess_images_count = list(itertools.chain.from_iterable(chess_images_path.glob(pattern) for pattern in ('*.jpg', '*.png')))
    # print(chess_images_count)
    for name in chess_images_count:
        chess_img = cv2.imread(str(name))
        chess_gray = cv2.cvtColor(chess_img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(chess_gray, (5,4), None)
        print(ret)
        if ret == True:
            real_points.append(obj_points)
            corners2 = cv2.cornerSubPix(chess_gray,corners, (11,11), (-1,-1), term_criteria)
            img_points.append(corners)
            cv2.drawChessboardCorners(chess_img, (7,6), corners2, ret)
            cv2.imshow('img', chess_img)
            cv2.waitKey(0)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(real_points, img_points, chess_gray.shape[::-1], None, None)

    return mtx, dist, rvecs, tvecs

## undistorting the camera
def undistort_camera(mtx, dist, rvecs, tvecs):
    img = cv2.imread('5.png')
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    cv2.imshow('Undistorted Image', dst)
    cv2.imshow('distorted image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    calbrate_distort_camera_path = Path("OS08A10_distorted_images")
    mtx, dist, rvecs, tvecs = calbrate_distorted_camera_based_on_images(calbrate_distort_camera_path)
    undistort_camera(mtx, dist, rvecs, tvecs)
