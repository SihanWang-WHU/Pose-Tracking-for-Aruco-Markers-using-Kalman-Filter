import numpy as np
import time
import cv2
import cv2.aruco as aruco
import math
from scipy.spatial.transform import Rotation as R

# 摄像机标定出来的参数
mtx = np.array([
    [1836.34160, 0, 1040.28052],
    [0, 1848.78821, 588.617205],
    [0, 0, 1], ])
dist = np.array([0.03673332, -1.08232878, 0.00832113, 0.00940459, 3.4011114])

# 定义视频打开的路径

video = "./6static_eclipse.mp4"
file_name = '6static_eclipse.txt'
file_name_kalman = '6static_eclipse_kf.txt'


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


# Aruco定位结果类
class Aruco_Res():
    # epoch表示为探测成功的历元，fail表示为探测失败的历元
    epoch = 0
    fail = 0
    markerSize = 6
    totalMarkers = 250
    markerlength = 0.083

    detect_or_not = 0

    covariance_x = 0.5
    covariance_z = 0.5

    # 视频流信息
    cap = cv2.VideoCapture(video)

    # time 为从视频中获取的时间戳, 单位为秒
    # dtime 为高斯马尔可夫过程的相关时间，默认为1/60（相邻帧之间的时间）
    time = 0
    dtime = 1 / 60

    # transform_translation用于表示Tvec中的平移量
    stat_transform_translation_x = 0
    stat_transform_translation_y = 0
    stat_transform_translation_z = 0

    # transform_rotation用于旋转的姿态四元数表达
    stat_transform_rotation_x = 0
    stat_transform_rotation_y = 0
    stat_transform_rotation_z = 0
    stat_transform_rotation_w = 0

    # transform_translation用于表示Tvec中的平移量
    transform_translation_x = 0
    transform_translation_y = 0
    transform_translation_z = 0

    # transform_rotation用于旋转的姿态四元数表达
    transform_rotation_x = 0
    transform_rotation_y = 0
    transform_rotation_z = 0
    transform_rotation_w = 0

    # transform_translation_dif 表示 transform_translation的导数
    stat_transform_translation_x_dif = 0
    stat_transform_translation_y_dif = 0
    stat_transform_translation_z_dif = 0

    # transform_rotation_dif 表示 transform_rotation的导数
    stat_transform_rotation_x_dif = 0
    stat_transform_rotation_y_dif = 0
    stat_transform_rotation_z_dif = 0
    stat_transform_rotation_w_dif = 0

    # 状态量和观测值的初始化
    # xt是待估状态 (14*1)
    # zt是观测值 (7*1)
    # omegat 表示 xt 的方差 (14*1)
    # vt 表示 zt 的方差 (7*1)
    H = np.empty([7, 14], dtype=float)
    A = np.empty([14, 14], dtype=float)
    xt = np.empty([14, 1], dtype=float)
    Pt = np.eye(14, dtype=float)
    zt = np.empty([7, 1], dtype=float)
    omegat = covariance_x * np.eye(14, dtype=float)
    vt = covariance_z * np.eye(7, dtype=float)
    x_pre = np.empty([14, 1], dtype=float)
    P_pre = np.eye(14, dtype=float)

    def __init__(self):
        pass

    def ArucoPositioning(self):

        font = cv2.FONT_HERSHEY_SIMPLEX  # font for displaying text (below)
        ret, frame = self.cap.read()
        # get time of the video (s)
        self.time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        # operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # aruco is a value
        # DICT_{markerSize}X{markerSize}_{totalMarkers} is a string
        key = getattr(aruco, f'DICT_{self.markerSize}X{self.markerSize}_{self.totalMarkers}')
        aruco_dict = aruco.Dictionary_get(key)
        parameters = aruco.DetectorParameters_create()

        # lists of ids and the corners belonging to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,
                                                              aruco_dict,
                                                              parameters=parameters)
        # 这里的corners是以像素为单位的

        #    if ids != None:
        if ids is not None:
            # 成功结算出来的历元数+1
            self.epoch += 1
            self.detect_or_not = 1
            rvecs, tvecs, markerpoints = aruco.estimatePoseSingleMarkers(corners, self.markerlength, mtx, dist)
            # Estimate pose of each marker and return the values rvet and tvec---different
            # from camera coefficients
            (rvecs - tvecs).any()  # get rid of that nasty numpy value array error

            # Print the pose for the ArUco marker
            # The pose of the marker is with respect to the camera lens frame.
            # Imagine you are looking through the camera viewfinder,
            # the camera lens frame's:
            # x-axis points to the right
            # y-axis points straight down towards your toes
            # z-axis points straight ahead away from your eye, out of the camera

            # Store the translation (i.e. position) information
            self.transform_translation_x = tvecs[0][0][0]
            self.transform_translation_y = tvecs[0][0][1]
            self.transform_translation_z = tvecs[0][0][2]

            # Store the rotation information
            rotation_matrix = np.eye(4)
            rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[0][0]))[0]
            r = R.from_matrix(rotation_matrix[0:3, 0:3])
            quat = r.as_quat()

            # Quaternion format
            self.transform_rotation_x = quat[0]
            self.transform_rotation_y = quat[1]
            self.transform_rotation_z = quat[2]
            self.transform_rotation_w = quat[3]

            # Euler angle format in radians
            roll_x, pitch_y, yaw_z = euler_from_quaternion(self.transform_rotation_x,
                                                           self.transform_rotation_y,
                                                           self.transform_rotation_z,
                                                           self.transform_rotation_w)

            roll_x = math.degrees(roll_x)
            pitch_y = math.degrees(pitch_y)
            yaw_z = math.degrees(yaw_z)
            #print("transform_translation_x: {}".format(self.transform_translation_x))
            #print("transform_translation_y: {}".format(self.transform_translation_y))
            #print("transform_translation_z: {}".format(self.transform_translation_z))
            #print("roll_x: {}".format(roll_x))
            #print("pitch_y: {}".format(pitch_y))
            #print("yaw_z: {}".format(yaw_z))
            #print()

            # 将计算出来的数据写入文件
            with open(file_name, 'a') as file_obj:
                file_obj.write('%f       %f       %f       %f       %f       %f       %f       '
                               % (self.time, self.transform_translation_x, self.transform_translation_y,
                                  self.transform_translation_z,
                                  roll_x, pitch_y, yaw_z))
                file_obj.write('\n')
                file_obj.close()

            # Draw the axes on the marker
            # 在画面上 标注aruco标签的各轴
            for i in range(rvecs.shape[0]):
                aruco.drawAxis(frame, mtx, dist, rvecs[i, :, :], tvecs[i, :, :], 0.03)
                aruco.drawDetectedMarkers(frame, corners)
            # DRAW ID
            cv2.putText(frame, "Id: " + str(ids), (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)


        else:
            ##### DRAW "NO IDS" #####
            self.fail += 1
            cv2.putText(frame, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.namedWindow("frame", 0)
        cv2.resizeWindow("frame", 1440, 720)  # 设置窗口大小
        cv2.imshow("frame", frame)

        key = cv2.waitKey(1)

        if key == 27:  # 按esc键退出
            print('esc break...')
            self.cap.release()
            cv2.destroyAllWindows()

        if key == ord(' '):  # 按空格键保存
            # num = num + 1
            # filename = "frames_%s.jpg" % num  # 保存一张图像
            filename = str(time.time())[:10] + ".jpg"
            cv2.imwrite(filename, frame)

    def Kalman_Filter(self):
        # 利用Kalman滤波来推算位置/量测更新
        # xˆk/k−1 = Φk/k−1xˆk−1
        # Pk/k−1 = Φk/k−1Pk−1ΦT k/k−1 + Γk−1Qk−1ΓT k−1
        # 构造H矩阵
        H_left = np.eye(7, dtype=float)
        H_right = np.zeros([7, 7], dtype=float)
        self.H = np.hstack((H_left, H_right))
        # 构造A矩阵
        A_eye = np.eye(7, dtype=float)
        A_dtime = self.dtime * np.eye(7, dtype=float)
        A_up = np.hstack((A_eye, A_dtime))
        A_zeros = np.zeros([7, 7], dtype=float)
        A_down = np.hstack((A_zeros, A_eye))
        self.A = np.vstack((A_up, A_down))

        if (self.epoch == 1):
            # 第一个历元只做一步预测
            self.xt = np.array([[self.transform_translation_x], [self.transform_translation_y],[self.transform_translation_z],
                       [self.transform_rotation_x], [self.transform_rotation_y], [self.transform_rotation_z],
                       [self.transform_rotation_w], [0], [0], [0], [0], [0], [0], [0]])
            self.Predict()
            # 赋值
            self.stat_transform_translation_x = self.x_pre[0, 0]
            self.stat_transform_translation_y = self.x_pre[1, 0]
            self.stat_transform_translation_z = self.x_pre[2, 0]
            self.stat_transform_rotation_x = self.x_pre[3, 0]
            self.stat_transform_rotation_y = self.x_pre[4, 0]
            self.stat_transform_rotation_z = self.x_pre[5, 0]
            self.stat_transform_rotation_w = self.x_pre[6, 0]
            self.stat_transform_translation_x_dif = self.x_pre[7, 0]
            self.stat_transform_translation_y_dif = self.x_pre[8, 0]
            self.stat_transform_translation_z_dif = self.x_pre[9, 0]
            self.stat_transform_rotation_x_dif = self.x_pre[10, 0]
            self.stat_transform_rotation_y_dif = self.x_pre[11, 0]
            self.stat_transform_rotation_z_dif = self.x_pre[12, 0]
            self.stat_transform_rotation_w_dif = self.x_pre[13, 0]
            return 0
        else:
            self.xt = np.array([[self.stat_transform_translation_x], [self.stat_transform_translation_y], [self.stat_transform_translation_z],
                                [self.stat_transform_rotation_x], [self.stat_transform_rotation_y], [self.stat_transform_rotation_z],
                                [self.stat_transform_rotation_w],  [self.stat_transform_translation_x_dif], [self.stat_transform_translation_y_dif],
                                [self.stat_transform_translation_z_dif], [self.stat_transform_rotation_x_dif], [self.stat_transform_rotation_y_dif],
                                [self.stat_transform_rotation_z_dif], [self.stat_transform_rotation_w_dif]])

            self.zt = np.array([[self.transform_translation_x, self.transform_translation_y,
                                 self.transform_translation_z, self.transform_rotation_x, self.transform_rotation_y,
                                 self.transform_rotation_z, self.transform_rotation_w]]).transpose()
            self.Predict()
            self.Update()
            # 结果的保存
            self.stat_transform_translation_x = self.xt[0, 0]
            self.stat_transform_translation_y = self.xt[1, 0]
            self.stat_transform_translation_z = self.xt[2, 0]
            self.stat_transform_rotation_x = self.xt[3, 0]
            self.stat_transform_rotation_y = self.xt[4, 0]
            self.stat_transform_rotation_z = self.xt[5, 0]
            self.stat_transform_rotation_w = self.xt[6, 0]
            self.stat_transform_translation_x_dif = self.xt[7, 0]
            self.stat_transform_translation_y_dif = self.xt[8, 0]
            self.stat_transform_translation_z_dif = self.xt[9, 0]
            self.stat_transform_rotation_x_dif = self.xt[10, 0]
            self.stat_transform_rotation_y_dif = self.xt[11, 0]
            self.stat_transform_rotation_z_dif = self.xt[12, 0]
            self.stat_transform_rotation_w_dif = self.xt[13, 0]
        self.Print()

    def Predict(self):
        self.x_pre = np.dot(self.A, self.xt)
        self.P_pre = np.dot(self.A, np.dot(self.Pt, self.A.transpose())) + self.omegat

    def Update(self):
        Mat1 = np.dot(self.H, np.dot(self.P_pre, self.H.transpose())) + self.vt
        Kk = np.dot(self.P_pre, np.dot(self.H.transpose(), np.linalg.inv(Mat1)))
        self.xt = self.x_pre + np.dot(Kk, (self.zt - np.dot(self.H, self.x_pre)))
        Mat2 = np.eye(14) - np.dot(Kk, self.H)
        self.Pt = np.dot(Mat2, np.dot(self.P_pre, Mat2.transpose())) \
                  + np.dot(Kk, np.dot(self.vt, Kk.transpose()))

    def Print(self):
        if(self.epoch == 1):
            return 0
        else:
            roll_x, pitch_y, yaw_z = euler_from_quaternion(self.stat_transform_rotation_x,
                                                           self.stat_transform_rotation_y,
                                                           self.stat_transform_rotation_z,
                                                           self.stat_transform_rotation_w)
            roll_x = math.degrees(roll_x)
            pitch_y = math.degrees(pitch_y)
            yaw_z = math.degrees(yaw_z)
            print("time is :{}".format(self.time))
            print("transform_translation_x: {}".format(self.stat_transform_translation_x))
            print("transform_translation_y: {}".format(self.stat_transform_translation_y))
            print("transform_translation_z: {}".format(self.stat_transform_translation_z))
            print("roll_x: {}".format(roll_x))
            print("pitch_y: {}".format(pitch_y))
            print("yaw_z: {}".format(yaw_z))
            print()
            # 将计算出来的数据写入文件
            with open(file_name_kalman, 'a') as file_obj_kalman:
                file_obj_kalman.write('%f       %f       %f       %f       %f       %f       %f       '
                                      % (self.time, self.stat_transform_translation_x,
                                         self.stat_transform_translation_y, self.stat_transform_translation_z,
                                         roll_x, pitch_y, yaw_z))
                file_obj_kalman.write('\n')
                file_obj_kalman.close()
            self.detect_or_not = 0

if __name__ == '__main__':
    # 打开一次文件并清空
    with open(file_name, 'w') as file_obj:
        file_obj.close()
    with open(file_name_kalman,'w') as file_obj_kalman:
        file_obj_kalman.close()

    # 进入滤波之前的准备工作、初始化
    Marker = Aruco_Res()

    while (True):
        Marker.ArucoPositioning()
        if(Marker.detect_or_not == 1):
            Marker.Kalman_Filter()