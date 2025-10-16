#!/usr/bin/env python3
import socket
import threading
import argparse
import time
import sys

def recv_loop(sock, stop_flag):
    sock.settimeout(1.0)  # 종료 체크를 위해 짧게 타임아웃
    while not stop_flag["stop"]:
        try:
            data, addr = sock.recvfrom(4096)
            print(f"[PY RECV] from {addr}: {data.decode(errors='ignore')}")
        except socket.timeout:
            continue
        except Exception as e:
            print(f"[PY RECV ERR] {e}")
            break

def send_loop(sock, peer_addr, stop_flag, interval=1.0):
    count = 0
    while not stop_flag["stop"]:
        msg = f"Hello from PY #{count}"
        try:
            sock.sendto(msg.encode(), peer_addr)
            print(f"[PY SEND] to {peer_addr}: {msg}")
        except Exception as e:
            print(f"[PY SEND ERR] {e}")
            break
        count += 1
        time.sleep(interval)

def main():
    parser = argparse.ArgumentParser(description="UDP duplex (Python)")
    parser.add_argument("--local-port", type=int, required=True, help="내가 bind할 포트")
    parser.add_argument("--peer-ip", type=str, required=True, help="상대 IP")
    parser.add_argument("--peer-port", type=int, required=True, help="상대 포트")
    args = parser.parse_args()

    # 소켓 생성 및 바인드(수신용)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.local_port))
    print(f"[PY] Listening on 0.0.0.0:{args.local_port}, peer={args.peer_ip}:{args.peer_port}")

    stop_flag = {"stop": False}
    peer_addr = (args.peer_ip, args.peer_port)

    t_recv = threading.Thread(target=recv_loop, args=(sock, stop_flag), daemon=True)
    t_send = threading.Thread(target=send_loop, args=(sock, peer_addr, stop_flag), daemon=True)
    t_recv.start()
    t_send.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[PY] Stopping...")
    finally:
        stop_flag["stop"] = True
        time.sleep(0.2)
        sock.close()
        print("[PY] Bye.")

if __name__ == "__main__":
    main()
