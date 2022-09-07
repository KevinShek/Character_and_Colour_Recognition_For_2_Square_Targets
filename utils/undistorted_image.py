import numpy as np
import cv2

## calbrating the distorted camera based on saved images for calbration 
def calbrate_distorted_camera_based_on_images():
    term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    obj_points = np.zeros((6*7,3), np.float32)
    obj_points[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    real_points = []
    img_points = []
    chess_images = [1,2,3,4,5,6,7,8,9,10]

    for name in chess_images:
        chess_img = cv2.imread(str(name)+'.png')
        chess_gray = cv2.cvtColor(chess_img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(chess_gray, (7,6), None)
        if ret == True:
            real_points.append(obj_points)
            corners2 = cv2.cornerSubPix(chess_gray,corners, (11,11), (-1,-1), term_criteria)
            img_points.append(corners)

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
