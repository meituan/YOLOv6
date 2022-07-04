import cv2
import torch
import numpy as np
import os

from .datasets import IMG_FORMATS
from .data_augment import letterbox
from ..utils.events import LOGGER

class DataSource:
    def __init__(self, img_size, stride, half):
        self.img_size = img_size
        self.stride = stride
        self.half = half

    def precess_image(self, img_src):
        '''Process image before image inference.'''
        image = letterbox(img_src, self.img_size, stride=self.stride)[0]

        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = torch.from_numpy(np.ascontiguousarray(image))
        image = image.half() if self.half else image.float()  # uint8 to fp16/32
        image /= 255  # 0 - 255 to 0.0 - 1.0

        return image, img_src


class ImagesSource(DataSource):
    def __init__(self, path, img_size, stride, half):
        super().__init__(img_size, stride, half)
        self.path = path
        self.img_paths = sorted(glob.glob(os.path.join(path, '*.*')))  # dir
        self.img_paths = [i for i in self.img_paths if i.split('.')[-1].lower() in IMG_FORMATS]
        self.current_path = self.img_paths[0]
        self.current = 0

    def __len__(self):
        return len(self.img_paths)

    def read(self):
        output = None, None
        self.current_path = self.img_paths[self.current]

        try:
            img_src = cv2.imread(self.img_paths[self.current])
            assert img_src is not None, f'Invalid image: {path}'
            output = self.precess_image(img_src)
        except Exception as e:
            LOGGER.warning(e)

        self.current += 1
        return output


class VideoSource(DataSource):
    def __init__(self, path, img_size, stride, half):
        super().__init__(img_size, stride, half)
        print(path)
        self.cap = cv2.VideoCapture(path)
        self.current_path = path.split('.')[0] + '-0.jpg'
        self.path = path

    def __len__(self):
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # In case of invalid video, this can append on network camera (RTSP protocol)
        # The camera failed to send the DTS (date time stamp) and PTS (?) segment
        # In such case, cv2 return a negative number.
        # Since the number of frames cannot be determined -> skip this source
        return max(0, frame_count)

    def read(self):
        ok, img = self.cap.read()

        frame_id = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.current_path = self.path.split('.')[0] + '-' + str(frame_id) + '.jpg'

        if ok:
            return self.precess_image(img)
        return None, None
