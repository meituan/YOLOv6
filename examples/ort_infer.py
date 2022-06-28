import cv2
import sys
sys.path.append('../')
import time
import random
import numpy as np
import onnxruntime as ort
from PIL import Image
from yolov6.data.data_augment import letterbox


names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']
colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}


cuda = False
w = "../weights/yolov6s.onnx"
image_path  = '../data/images/image1.jpg'

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
session = ort.InferenceSession(w, providers=providers)

image = cv2.imread(image_path)
image,ratio,dwdh = letterbox(image,auto=False)
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

image_copy = image.copy()

image = image.transpose((2, 0, 1))
image = np.expand_dims(image,0)
image = np.ascontiguousarray(image)
im = image.astype(np.float32)
im/=255


# warmup for 10 times
for _ in range(10):
    outputs = session.run([i.name for i in session.get_outputs()], {'image_arrays': im})[0]

start = time.perf_counter()
outputs = session.run([i.name for i in session.get_outputs()], {'image_arrays': im})[0]
print(f'Cost {time.perf_counter()-start} s')


for i,(batch,x0,y0,x1,y1,clas,score) in enumerate(outputs):
    name = names[int(clas)]
    color = colors[name]
    cv2.rectangle(image_copy,[int(x0.round()),int(y0.round())],[int(x1.round()),int(y1.round())],color,2)
    cv2.putText(image_copy,name,(int(x0.round()), int(y0.round())-2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)

Image.fromarray(image_copy).show()

