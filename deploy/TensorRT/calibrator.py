import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import glob
from tensorrt_processor import letterbox

import ctypes
import logging
logger = logging.getLogger(__name__)
ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_char_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]


"""
There are 4 types calibrator in TensorRT.
trt.IInt8LegacyCalibrator
trt.IInt8EntropyCalibrator
trt.IInt8EntropyCalibrator2
trt.IInt8MinMaxCalibrator
"""

IMG_FORMATS = [".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".dng", ".webp", ".mpo"]
IMG_FORMATS.extend([f.upper() for f in IMG_FORMATS])

class Calibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, stream, cache_file=""):
        trt.IInt8MinMaxCalibrator.__init__(self)
        self.stream = stream
        self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)
        self.cache_file = cache_file
        stream.reset()

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self, names):
        print("######################")
        print(names)
        print("######################")
        batch = self.stream.next_batch()
        if not batch.size:
            return None

        cuda.memcpy_htod(self.d_input, batch)
        return [int(self.d_input)]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                logger.info("Using calibration cache to save time: {:}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            logger.info("Caching calibration data for future use: {:}".format(self.cache_file))
            f.write(cache)


def process_image(img_src, img_size, stride):
    '''Process image before image inference.'''
    image = letterbox(img_src, img_size, auto=False)[0]
    # Convert
    image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    image = np.ascontiguousarray(image).astype(np.float32)
    image /= 255.  # 0 - 255 to 0.0 - 1.0
    return image

class DataLoader:
    def __init__(self, batch_size, batch_num, calib_img_dir, input_w, input_h):
        self.index = 0
        self.length = batch_num
        self.batch_size = batch_size
        self.input_h = input_h
        self.input_w = input_w
        # self.img_list = [i.strip() for i in open('calib.txt').readlines()]
        self.img_list = [os.path.join(calib_img_dir, x) for x in os.listdir(calib_img_dir) if os.path.splitext(x)[-1] in IMG_FORMATS]
        assert len(self.img_list) > self.batch_size * self.length, \
            '{} must contains more than '.format(calib_img_dir) + str(self.batch_size * self.length) + ' images to calib'
        print('found all {} images to calib.'.format(len(self.img_list)))
        self.calibration_data = np.zeros((self.batch_size, 3, input_h, input_w), dtype=np.float32)

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.length:
            for i in range(self.batch_size):
                assert os.path.exists(self.img_list[i + self.index * self.batch_size]), f'{self.img_list[i + self.index * self.batch_size]} not found!!'
                img = cv2.imread(self.img_list[i + self.index * self.batch_size])
                img = process_image(img, [self.input_h, self.input_w], 32)

                self.calibration_data[i] = img

            self.index += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def __len__(self):
        return self.length
