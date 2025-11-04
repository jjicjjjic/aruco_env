# 1. 'jh_control' 폴더 안에 있는 모든 .cpp 파일의 목록을 찾아서 'CPP_FILES'라는 변수에 저장합니다.
file(GLOB CPP_FILES "kjh/*.cpp")

# 2. 찾은 파일 목록을 하나씩 순회하면서 반복합니다.
foreach(CPP_FILE ${CPP_FILES})
    # 3. 파일 경로에서 확장자(.cpp)를 제외한 순수 파일 이름만 추출하여 'EXE_NAME' 변수에 저장합니다.
    # 예: "jh_control/jh.cpp" -> "jh"
    get_filename_component(EXE_NAME ${CPP_FILE} NAME_WE)

    # 4. 추출한 파일 이름으로 실행 파일을 만듭니다.
    add_executable(${EXE_NAME} ${CPP_FILE})

    # 5. 해당 실행 파일에 rbpodo 라이브러리를 연결합니다.
    target_link_libraries(${EXE_NAME} rbpodo::rbpodo)
endforeach()


cmake_minimum_required(VERSION 3.15)
if (NOT DEFINED SKBUILD_PROJECT_VERSION)
    file(READ ${CMAKE_SOURCE_DIR}/pyproject.toml PYPROJECT_STR)
    string(REGEX MATCHALL "[^\n\r]+" PYPROJECT_STR_LIST "${PYPROJECT_STR}")

    foreach (STR ${PYPROJECT_STR_LIST})
        if (${STR} MATCHES "^version[ \t\r\n]+=[ \t\r\n]+\"(.*)\"")
            set(RBPODO_VERSION ${CMAKE_MATCH_1})
        endif ()
    endforeach ()
else ()
    set(RBPODO_VERSION ${SKBUILD_PROJECT_VERSION})
endif ()
project(rbpodo
        VERSION ${RBPODO_VERSION}
        LANGUAGES CXX)
message(STATUS "rbpodo project version: ${RBPODO_VERSION}")

if (NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(CMAKE_CXX_STANDARD 17)
set(BUILD_EXAMPLES OFF CACHE BOOL "Build examples")
set(BUILD_EIGEN_EXAMPLES OFF CACHE BOOL "Build examples (Eigen)")
set(BUILD_PYTHON_BINDINGS OFF CACHE BOOL "Build Python bindings using Pybind11")

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.24)
    cmake_policy(SET CMP0135 OLD)
endif ()

if (BUILD_PYTHON_BINDINGS)
    find_package(Eigen3 CONFIG QUIET)
    if (Eigen3_FOUND)
        message(STATUS "Found Eigen3: ${Eigen3_DIR}")
    else ()
        message(WARNING "Not found Eigen3")
        message(STATUS "Downloading Eigen3 ...")
        include(FetchContent)
        set(EXTERNALS_DIR ${CMAKE_BINARY_DIR}/external)
        FetchContent_Declare(libeigen
                URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2
                URL_MD5 132dde48fe2b563211675626d29f1707
                SOURCE_DIR ${EXTERNALS_DIR}/eigen
                EXCLUDE_FROM_ALL
        )
        
        set(BUILD_TESTING OFF CACHE BOOL "")
        message(STATUS "fetch content make available")
        FetchContent_MakeAvailable(libeigen)
        message(STATUS "done")
        unset(BUILD_TESTING CACHE)
    endif ()
endif ()


add_subdirectory(src)
if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()
if (BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif ()

## Installation

if (NOT BUILD_PYTHON_BINDINGS)
    include(GNUInstallDirs)
    set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/rbpodo)

    install(TARGETS rbpodo
            EXPORT rbpodo-targets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

    set_target_properties(rbpodo PROPERTIES EXPORT_NAME rbpodo)

    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    install(EXPORT rbpodo-targets
            FILE
            rbpodo-targets.cmake
            NAMESPACE
            rbpodo::
            DESTINATION
            ${INSTALL_CONFIGDIR})

    include(CMakePackageConfigHelpers)
    configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/cmake/rbpodo-config.cmake.in
            ${CMAKE_CURRENT_BINARY_DIR}/rbpodo-config.cmake
            INSTALL_DESTINATION ${INSTALL_CONFIGDIR})
    install(FILES
            ${CMAKE_CURRENT_BINARY_DIR}/rbpodo-config.cmake
            DESTINATION ${INSTALL_CONFIGDIR})
endif ()


