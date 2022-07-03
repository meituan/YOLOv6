cmake_minimum_required(VERSION 3.10)
project(yolov6)

set(CMAKE_CXX_STANDARD 11)

if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-std=c++11 -Wno-deprecated ${CMAKE_CXX_FLAGS} ")
    message(STATUS "[Linux GCC Compiler Options]+:-std=c++11 -Wno-deprecated")
endif ()

set(YOLOv6_ROOT_DIR ${CMAKE_SOURCE_DIR})
set(BUILD_YOLOv6_DIR ${YOLOv6_ROOT_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${BUILD_YOLOv6_DIR})

# check platform
message(STATUS "Checking build platform for: ${CMAKE_CURRENT_SOURCE_DIR}")

if(WIN32)
    set(PLATFORM_NAME windows)
    message(FATAL_ERROR "This YOLOv6 C++ demo does not support for Windows now!")
elseif(APPLE)
    set(PLATFORM_NAME macos)
elseif(UNIX AND NOT APPLE AND NOT ANDROID)
    set(PLATFORM_NAME linux)
elseif (ANDROID)
    set(PLATFORM_NAME android)  # armeabi-v7a | arm64-v8a
    message(FATAL_ERROR "This YOLOv6 C++ demo does not support for Android now!")
else()
    message(FATAL_ERROR "This YOLOv6 C++ demo does not support for ${CMAKE_SYSTEM_NAME} now!")
endif()
message(STATUS "Checking build platform done!: [${PLATFORM_NAME}]")

# setup include & lib paths
include_directories(${YOLOv6_ROOT_DIR}/src)
include_directories(${YOLOv6_ROOT_DIR}/third_party/include)
link_directories(${YOLOv6_ROOT_DIR}/third_party/lib/${PLATFORM_NAME})

# setup opencv
message(STATUS "Setting up OpenCV libs for: ${CMAKE_CURRENT_SOURCE_DIR}")

if (NOT WIN32)
    set(OpenCV_LIBS
            opencv_core
            opencv_imgproc
            opencv_imgcodecs) # no videoio, video module
else ()
    set(OpenCV_LIBS opencv_world452)
endif ()

message(STATUS "Setting up OpenCV libs done! OpenCV_LIBS:+[${OpenCV_LIBS}]")

# add target
add_executable(yolov6 main.cpp src/utils.cpp src/handler.cpp src/yolov6.impl.cpp)
target_link_libraries(yolov6 ${OpenCV_LIBS} ncnn)  # link opencv and NCNN
message(STATUS "added yolov6 executable!")

# install deps to exe path
message("Installing OpenCV libs ...")
file(GLOB ALL_OpenCV_LIBS ${YOLOv6_ROOT_DIR}/third_party/lib/${PLATFORM_NAME}/*opencv*)
file(INSTALL ${ALL_OpenCV_LIBS} DESTINATION ${EXECUTABLE_OUTPUT_PATH})
message("Installing NCNN libs ...")
file(GLOB ALL_NCNN_LIBS ${YOLOv6_ROOT_DIR}/third_party/lib/${PLATFORM_NAME}/*ncnn*)
file(INSTALL ${ALL_NCNN_LIBS} DESTINATION ${EXECUTABLE_OUTPUT_PATH})
message("Installed libs !")
