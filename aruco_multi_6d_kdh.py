import pyrealsense2 as rs
import cv2
import numpy as np
import math
import time
import os

MARKER_LENGTH_M = 0.030
DURATION_SEC = 5  # ★ 실행 시간(초) 설정: 원하는 시간으로 변경 가능

# ArUco 설정
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# 카메라 파이프라인 구성
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)
color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr = color_stream.get_intrinsics()
camera_matrix = np.array([[intr.fx, 0, intr.ppx],
                          [0, intr.fy, intr.ppy],
                          [0,      0,      1]], dtype=np.float64)
dist_coeffs = np.array(intr.coeffs[:5], dtype=np.float64)

align = rs.align(rs.stream.color)

def rotation_matrix_to_rpy(R):
    roll  = math.atan2(R[2,1], R[2,2])
    pitch = math.atan2(-R[2,0], math.sqrt(R[2,1]**2 + R[2,2]**2))
    yaw   = math.atan2(R[1,0], R[0,0])
    return np.degrees([roll, pitch, yaw])

try:
    start_time = time.time()  # ★ 시작 시간 기록

    while True:
        elapsed = time.time() - start_time
        if elapsed > DURATION_SEC:  # ★ DURATION_SEC 초 경과 시 종료
            print(f"{DURATION_SEC}초 경과 - 프로그램 자동 종료")
            break

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        corners, ids, _ = detector.detectMarkers(color_image)

        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _objPts = cv2.aruco.estimatePoseSingleMarkers(
                corners, MARKER_LENGTH_M, camera_matrix, dist_coeffs
            )

            for corner, marker_id, rvec, tvec in zip(corners, ids.flatten(), rvecs, tvecs):
                pts = corner[0].astype(int)
                center = np.mean(pts, axis=0).astype(int)
                x, y = int(center[0]), int(center[1])
                z_depth = depth_frame.get_distance(x, y)

                cv2.polylines(color_image, [pts], True, (0, 255, 255), 2)
                cv2.circle(color_image, (x, y), 5, (0, 0, 255), -1)

                rvec = np.squeeze(rvec)
                tvec = np.squeeze(tvec)

                R, _ = cv2.Rodrigues(rvec)
                roll_deg, pitch_deg, yaw_deg = rotation_matrix_to_rpy(R)

                cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH_M)

                cv2.putText(color_image, f"ID:{marker_id}", (x-20, y-40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
                cv2.putText(color_image, f"Z(depth): {z_depth:.3f} m", (x-20, y-23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(color_image, f"RPY(deg): {roll_deg:.1f}, {pitch_deg:.1f}, {yaw_deg:.1f}", (x-20, y-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 2)

                tx, ty, tz = tvec.flatten()
                print(f"[ID:{marker_id}] tvec(m)={tx:.3f},{ty:.3f},{tz:.3f} | "
                      f"RPY(deg)={roll_deg:.1f},{pitch_deg:.1f},{yaw_deg:.1f}")

        cv2.imshow('RealSense ArUco Multi-Tracking', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("사용자 종료 요청(q)")
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
