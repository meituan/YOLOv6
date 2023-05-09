# ncnn-android-yolov6

The YOLOv6 object detection demo of `Android`.
You can directly download apk file from [Android Demo here](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6-android-demo.apk), many thanks to [triple Mu](https://github.com/triple-Mu).

This is a sample ncnn android project, it depends on ncnn library and opencv

- [ncnn](https://github.com/Tencent/ncnn)

- [opencv-mobile](https://github.com/nihui/opencv-mobile)


## How to build and run
### step1

* Download [ncnn-YYYYMMDD-android-vulkan.zip](https://github.com/Tencent/ncnn/releases) or build ncnn for android yourself
* Extract `ncnn-YYYYMMDD-android-vulkan.zip` into `app/src/main/jni` and change the `ncnn_DIR` path to yours in `app/src/main/jni/CMakeLists.txt`

### step2

* Download [opencv-mobile-XYZ-android.zip](https://github.com/nihui/opencv-mobile)
* Extract `opencv-mobile-XYZ-android.zip` into `app/src/main/jni` and change the `OpenCV_DIR` path to yours in `app/src/main/jni/CMakeLists.txt`

### step3
* download [AndroidAssets.zip
](https://github.com/meituan/YOLOv6/releases/download/0.4.0/AndroidAssets.zip)
* Unzip `AndroidAssets.zip`, you will get a directory named as `assets`, move it
into `app/src/`.

### step4
* Open this project with Android Studio, build it and enjoy!

## some notes
* Android ndk camera is used for best efficiency
* Crash may happen on very old devices for lacking HAL3 camera interface
* All models are manually modified to accept dynamic input shape
* Most small models run slower on GPU than on CPU, this is common
* FPS may be lower in dark environment because of longer camera exposure time

## Reference：
- [ncnn-android-nanodet](https://github.com/nihui/ncnn-android-nanodet)
- [ncnn](https://github.com/Tencent/ncnn)
