#!/usr/bin/env python3
import socket
import threading
import argparse
import time
import json
import math
import cv2
import numpy as np
import pyrealsense2 as rs

MARKER_LENGTH_M = 0.030  # 한 변 길이 (미터)

# -------- ArUco 준비 --------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

def rotation_matrix_to_rpy(R):
    # R: 3x3 회전행렬, RPY 순서: X(roll)→Y(pitch)→Z(yaw), 라디안을 도로 변환
    roll  = math.atan2(R[2,1], R[2,2])
    pitch = math.atan2(-R[2,0], math.sqrt(R[2,1]**2 + R[2,2]**2))
    yaw   = math.atan2(R[1,0], R[0,0])
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]

def recv_loop(sock, stop_flag):
    sock.settimeout(1.0)
    while not stop_flag["stop"]:
        try:
            data, addr = sock.recvfrom(4096)
            print(f"[PY RECV] from {addr}: {data.decode(errors='ignore')}")
        except socket.timeout:
            continue
        except Exception as e:
            print(f"[PY RECV ERR] {e}")
            break

def main():
    parser = argparse.ArgumentParser(description="Send ArUco 6D pose over UDP (Python)")
    parser.add_argument("--local-port", type=int, required=True, help="내가 bind할 UDP 포트 (수신용)")
    parser.add_argument("--peer-ip", type=str, required=True, help="상대 IP (C++ 쪽)")
    parser.add_argument("--peer-port", type=int, required=True, help="상대 UDP 포트")
    parser.add_argument("--marker-id", type=int, default=None,
                        help="특정 ID만 전송 (미지정 시 감지된 모든 마커 전송)")
    parser.add_argument("--draw", action="store_true",
                        help="OpenCV 윈도우로 시각화 (원하면 사용)")
    args = parser.parse_args()

    # -------- UDP 소켓 --------
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.local_port))
    peer_addr = (args.peer_ip, args.peer_port)
    print(f"[PY] UDP listening on 0.0.0.0:{args.local_port}, peer={peer_addr}")

    stop_flag = {"stop": False}
    th_recv = threading.Thread(target=recv_loop, args=(sock, stop_flag), daemon=True)
    th_recv.start()

    # -------- RealSense 파이프라인 --------
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()  # rs.intrinsics
    camera_matrix = np.array([[intr.fx, 0, intr.ppx],
                              [0, intr.fy, intr.ppy],
                              [0,      0,      1]], dtype=np.float64)
    dist_coeffs = np.array(intr.coeffs[:5], dtype=np.float64)

    align = rs.align(rs.stream.color)

    print("[PY] RealSense started. Press 'q' in window or Ctrl+C in terminal to stop.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())

            # ---- ArUco 감지 ----
            corners, ids, _ = detector.detectMarkers(color_img)

            if ids is not None and len(ids) > 0:
                # 자세 추정 (카메라 좌표계 기준)
                rvecs, tvecs, _obj = cv2.aruco.estimatePoseSingleMarkers(
                    corners, MARKER_LENGTH_M, camera_matrix, dist_coeffs
                )

                # 각 마커별로 UDP로 JSON 전송
                for corner, marker_id, rvec, tvec in zip(corners, ids.flatten(), rvecs, tvecs):
                    if args.marker_id is not None and marker_id != args.marker_id:
                        continue  # 특정 ID 필터링

                    # 회전/병진 벡터 정리
                    rvec = np.squeeze(rvec)
                    tvec = np.squeeze(tvec)
                    R, _ = cv2.Rodrigues(rvec)
                    roll_deg, pitch_deg, yaw_deg = rotation_matrix_to_rpy(R)

                    # 마커 중앙 픽셀 위치 (깊이 확인용)
                    pts = corner[0].astype(int)
                    center = np.mean(pts, axis=0).astype(int)
                    cx, cy = int(center[0]), int(center[1])
                    z_depth_m = depth_frame.get_distance(cx, cy)

                    # 전송 payload (필요시 key 이름은 C++ 수신측에 맞추세요)
                    payload = {
                        "type": "aruco_6d",
                        "id": int(marker_id),
                        "timestamp": time.time(),
                        "tvec_m": {  # 카메라 좌표계 [m]
                            "x": float(tvec[0]),
                            "y": float(tvec[1]),
                            "z": float(tvec[2]),
                        },
                        "rpy_deg": {
                            "roll":  float(roll_deg),
                            "pitch": float(pitch_deg),
                            "yaw":   float(yaw_deg),
                        },
                        "depth_m": float(z_depth_m),  # 레이저 깊이 참조(카메라 중앙 픽셀 기준)
                    }

                    # UDP 전송
                    try:
                        msg = json.dumps(payload)
                        sock.sendto(msg.encode("utf-8"), peer_addr)
                        print(f"[PY SEND] {payload}")
                    except Exception as e:
                        print(f"[PY SEND ERR] {e}")

                    # ----- 시각화 옵션 -----
                    if args.draw:
                        cv2.polylines(color_img, [pts], True, (0, 255, 255), 2)
                        cv2.drawFrameAxes(color_img, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH_M)
                        cv2.putText(color_img, f"ID:{marker_id}", (cx-20, cy-40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
                        cv2.putText(color_img, f"Z(depth): {z_depth_m:.3f} m", (cx-20, cy-23),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        cv2.putText(color_img,
                                    f"RPY(deg): {roll_deg:.1f}, {pitch_deg:.1f}, {yaw_deg:.1f}",
                                    (cx-20, cy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 2)

            # 윈도우 표시
            if args.draw:
                cv2.imshow("RealSense ArUco 6D -> UDP", color_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n[PY] KeyboardInterrupt")

    finally:
        stop_flag["stop"] = True
        time.sleep(0.2)
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        sock.close()
        print("[PY] Bye.")
        
if __name__ == "__main__":
    main()
