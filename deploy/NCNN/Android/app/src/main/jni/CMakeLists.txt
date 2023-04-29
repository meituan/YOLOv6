project(yolov6ncnn)

cmake_minimum_required(VERSION 3.10)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.6.0-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20230223-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(yolov6ncnn SHARED yolov6ncnn.cpp yolo.cpp ndkcamera.cpp)

target_link_libraries(yolov6ncnn ncnn ${OpenCV_LIBS} camera2ndk mediandk)
