import pyrealsense2 as rs
import cv2
import numpy as np

# ArUco 설정
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# 카메라 파이프라인 구성
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# 정렬 객체: depth를 color에 정렬
align = rs.align(rs.stream.color)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # ArUco 마커 인식
        corners, ids, _ = detector.detectMarkers(color_image)

        if ids is not None:
            for corner, marker_id in zip(corners, ids.flatten()):
                pts = corner[0].astype(int)
                center = np.mean(pts, axis=0).astype(int)
                x, y = center
                z = depth_frame.get_distance(x, y)

                # 마커 외곽선 그리기
                cv2.polylines(color_image, [pts], True, (0, 255, 255), 2)
                # 중심점 시각화
                cv2.circle(color_image, (x, y), 5, (0, 0, 255), -1)
                # ID 및 거리 텍스트 표시
                cv2.putText(color_image, f"ID:{marker_id}", (x - 20, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
                cv2.putText(color_image, f"Z: {z:.3f}m", (x - 20, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                print(f"[ID:{marker_id}] x={x}, y={y}, z={z:.3f} m")

        # 결과 출력
        cv2.imshow('RealSense ArUco Multi-Tracking', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
