#include <iostream>
#include <fstream>
#include <array>
#include <thread>
#include <chrono>
#include <atomic>
#include <future>
#include "rbpodo/rbpodo.hpp"

using namespace rb;

// 비동기로 입력을 감지하는 함수
bool check_quit() {
  std::string input;
  std::getline(std::cin, input);
  return (input == "q" || input == "Q");
}

int main() {
  try {
    podo::Cobot robot("10.0.2.7");
    podo::ResponseCollector rc;

    std::ofstream file("/home/nrel/aruco/rbpodo/jh_control/data/tcp_pose_log.csv");
    if (!file.is_open()) {
      std::cerr << "파일을 열 수 없습니다." << std::endl;
      return 1;
    }

    file << "Time(ms),X(mm),Y(mm),Z(mm),Roll(deg),Pitch(deg),Yaw(deg)\n";

    std::cout << "TCP 자세 기록을 시작합니다. 종료하려면 'q'를 입력 후 Enter를 누르세요.\n";

    // 비동기 입력 감시 시작
    std::future<bool> quit_future = std::async(std::launch::async, check_quit);

    while (quit_future.wait_for(std::chrono::milliseconds(1)) == std::future_status::timeout) {
      std::array<double, 6> tcp_pose{};
      robot.get_tcp_info(rc, tcp_pose);
      rc.error().throw_if_not_empty();

      // 시간(ms)
      auto now = std::chrono::steady_clock::now().time_since_epoch();
      long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();

      // 파일에 저장
      file << ms << ",";
      for (int j = 0; j < 6; ++j) {
        file << tcp_pose[j];
        if (j < 5) file << ",";
      }
      file << "\n";

      std::this_thread::sleep_for(std::chrono::milliseconds(100));  // 10Hz 기록
    }

    file.close();
    std::cout << "'q' 입력 감지됨. TCP 자세값 기록을 종료합니다.\n";

  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}
