#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include <thread>
#include <chrono>
#include "rbpodo/rbpodo.hpp"

using namespace rb;

int main() {
  try {
    // (1) 로봇 연결
    podo::Cobot robot("10.0.2.7");
    podo::ResponseCollector rc;

    // (2) 시뮬레이션 모드 (실기일 경우 Real로 변경)
    robot.set_operation_mode(rc, podo::OperationMode::Simulation);
    rc.error().throw_if_not_empty();

    // (3) 속도 설정
    robot.set_speed_bar(rc, 0.3);
    robot.flush(rc);

    // (4) CSV 파일 열기
    std::ifstream file("/home/nrel/aruco/rbpodo/jh_control/data/tcp_pose_log.csv");
    if (!file.is_open()) {
      std::cerr << "CSV 파일을 열 수 없습니다!" << std::endl;
      return 1;
    }

    std::string line;
    bool first_line = true;

    while (std::getline(file, line)) {
      if (first_line) { 
        first_line = false; 
        continue;
      }

      std::stringstream ss(line);
      std::string token;
      std::array<double, 6> tcp_pose{};
      int idx = 0;
      std::getline(ss, token, ',');
      while (std::getline(ss, token, ',') && idx < 6) {
        tcp_pose[idx++] = std::stod(token);
      }

      if (idx == 6) {
        std::cout << "Moving to TCP pose: [";
        for (int i = 0; i < 6; ++i) {
          std::cout << tcp_pose[i];
          if (i < 5) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        robot.move_servo_l(rc, tcp_pose, 0.01, 0.05, 1, 0.05);
        rc.error().throw_if_not_empty();

      }
    }

    std::cout << "CSV 파일 기반 궤적 재생 완료!" << std::endl;

  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}
