# aruco_sender.py
import pyrealsense2 as rs
import cv2
import numpy as np
import socket

# --- 설정 ---
ROBOT_IP = "127.0.0.1"
ROBOT_PORT = 12345
MARKER_LENGTH_M = 0.030

# --- 초기화 ---
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr = color_stream.get_intrinsics()
camera_matrix = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)
dist_coeffs = np.array(intr.coeffs, dtype=np.float64)

print(f"UDP 데이터를 {ROBOT_IP}:{ROBOT_PORT}로 전송 시작...")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame: continue
        color_image = np.asanyarray(color_frame.get_data())

        corners, ids, _ = detector.detectMarkers(color_image)

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH_M, camera_matrix, dist_coeffs)
            
            # 첫 번째로 탐지된 마커 정보만 전송
            marker_id = ids.flatten()[0]
            rvec = rvecs[0].flatten()
            tvec = tvecs[0].flatten()

            # 데이터 포맷팅: "ID,tvec_x,tvec_y,tvec_z,rvec_x,rvec_y,rvec_z"
            data_string = f"{marker_id},{tvec[0]:.4f},{tvec[1]:.4f},{tvec[2]:.4f}," \
                          f"{rvec[0]:.4f},{rvec[1]:.4f},{rvec[2]:.4f}"

            # UDP로 "카메라 기준" 좌표 전송
            sock.sendto(data_string.encode(), (ROBOT_IP, ROBOT_PORT))
            
            cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH_M)
            cv2.putText(color_image, f"Sent: {data_string}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Python ArUco Sender", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    pipeline.stop()
    sock.close()
    cv2.destroyAllWindows()