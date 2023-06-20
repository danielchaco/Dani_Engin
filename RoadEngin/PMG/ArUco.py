import cv2
import numpy as np
import matplotlib.pyplot as plt

ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
SQUARE_SIZE = 0.300
MARKER_SIZE = 0.233
AURCO_PARAMS = cv2.aruco.DetectorParameters_create()
BOARD = cv2.aruco.CharucoBoard_create(3, 3, SQUARE_SIZE, MARKER_SIZE, ARUCO_DICT)
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
CRITERIA = (cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 1000, 1e-9)


def get_ArUcos_corners_ids(img):
    corners, ids, rejected = cv2.aruco.detectMarkers(img, ARUCO_DICT, parameters=AURCO_PARAMS)
    return corners, ids

def get_chessboard_corners_ids(img,aru_corners,aru_ids):
    """
    based on cv2.aruco.interpolateCornersCharuco method
    :param img:
    :return: chessboard corners.
    """
    res = cv2.aruco.interpolateCornersCharuco(aru_corners, aru_ids, img, BOARD)
    return res[1], res[2]

def get_params(img2D_size,corners,ids):
    ret, mtx, dist, rvecs, tvecs, _, _, _ = cv2.aruco.calibrateCameraCharucoExtended(
        charucoCorners=[corners] if type(corners) != list else corners,
        charucoIds=[ids] if type(ids) != list else ids,
        board=BOARD,
        imageSize=img2D_size,
        cameraMatrix=np.array([
            [472, 0, 960],
            [0, 476, 540],
            [0, 0, 1]
        ], dtype=np.float64),
        distCoeffs=np.zeros((5, 1)),
        flags=cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL,
        criteria=CRITERIA
    )
    return ret, mtx, dist, rvecs, tvecs


def img_scale(img, scale=0.5):
    return cv2.resize(img, (0, 0), fx=scale, fy=scale)


def plot_axis(img, aru_corners, aru_ids, mtx, dist, plot=True):
    img = cv2.aruco.drawDetectedMarkers(img, aru_corners, aru_ids)
    rvecs, tvecs, trash = cv2.aruco.estimatePoseSingleMarkers(aru_corners, MARKER_SIZE, mtx, dist)
    for r, t in zip(rvecs, tvecs):
        img = cv2.aruco.drawAxis(img, mtx, dist, r, t, SQUARE_SIZE)
    if plot:
        cv2.imshow('Axis', img_scale(img,0.7))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img


paths = [r'C:\Users\Daniel\PycharmProjects\Dani_Engin\RoadEngin\PMG\CalBoard\vlcsnap-2023-06-15-14h28m49s066.png',
         r'C:\Users\Daniel\PycharmProjects\Dani_Engin\RoadEngin\PMG\CalBoard\vlcsnap-2023-06-15-14h21m26s716.png']

img = cv2.imread(paths[0])

aru_corners, aru_ids = get_ArUcos_corners_ids(img)
mid_corners, mid_ids = get_chessboard_corners_ids(img, aru_corners,aru_ids)
ret, mtx, dist, rvecs, tvecs = get_params(img.shape[:2],mid_corners,mid_ids)
img2 = plot_axis(img, aru_corners, aru_ids, mtx, dist, plot=True)



#
# # cv2.imshow('Original half resized', cv2.resize(img, (0, 0), fx=0.5, fy=0.5))
# # cv2.imshow('5x5 board', BOARD.draw((500, 500)))
# #
# # # res = cv2.aruco.detectMarkers(gray, ARUCO_DICT)
# #
# # if len(res[0]) > 0:
# #     res2 = cv2.aruco.interpolateCornersCharuco(res[0], res[1], gray, board)
# #     if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3:
# #         corners2 = res2[1]
# #         Ids2 = res2[2]
# #
# #         img_size = gray.shape
# #         mtx_seed = np.array([
# #             [2000, 0, img_size[0] / 2],
# #             [0, 2000, img_size[1] / 2],
# #             [0, 0, 1]
# #         ])
# #         dist_coef_ini = np.zeros((5, 1))
# #         flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL)
# #         ret, mtx, dist, rvecs, tvecs, _, _, _ = cv2.aruco.calibrateCameraCharucoExtended(
# #             charucoCorners=[corners2],
# #             charucoIds=[Ids2],
# #             board=BOARD,
# #             imageSize=img_size,
# #             cameraMatrix=mtx_seed,
# #             distCoeffs=dist_coef_ini,
# #             flags=flags,
# #             criteria=criteria
# #         )
# #
# #         frame_markers = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)
# #         cv2.imshow('Detections', frame_markers)
# #
# #         #
# #         rvecs, tvecs, trash = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, mtx, dist)
# #         for r, t in zip(rvecs, tvecs):
# #             frame_markers = cv2.aruco.drawAxis(frame_markers, mtx, dist, r, t, SQUARE_SIZE)
# #
# #         cv2.imshow('Axis', frame_markers[500:, 300:-300])
# #
# #         # # print(corners)
# #         # mid_points = np.float32([[np.mean(corner.reshape((4, 2))[:, 0]), np.mean(corner.reshape((4, 2))[:, 1])] for corner in corners])
# #         # eq_points = np.float32([[0,300],[300,0],[300,600],[600,300]])
# #         #
# #         # M = cv2.getPerspectiveTransform(mid_points,eq_points)
# #         # dst = cv2.warpPerspective(img,M,(600,600))
# #         # cv2.imshow('Transform',dst)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # # Camera calibration
# # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# # didn't work
# #
# # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # cv2.imshow('gray',gray)
# # ret, corners = cv2.findChessboardCorners(gray, (4,4), None)
# # print(ret,corners)
# # imgpoints = []
# #
# # if ret == True:
# #     print('yes sr')
# #     corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
# #     imgpoints.append(corners2)
# #     #display
# #     cor = cv2.drawChessboardCorners(img.copy(),(3,3),corners2,ret)
# #     cv2.imshow('corners',cor)
