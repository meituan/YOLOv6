# 模型转换
### 此文档将展示如何把 在3.0代码库下训练出来的模型 转换为 可在4.0代码库训练测试的模型。
### 具体步骤如下：

### 0. 分别拉取3.0代码库和4.0代码库

### 1. 基于4.0代码库下训练相同配置的模型，当运行一个epoch后会在保存目录下保存模型文件，即可停止训练
通过 `torch.load`加载模型文件，并将模型里所有权重的键值 `model.keys()` 保存下来.

### 2. 在3.0代码库下进行模型权重键值替换
其中newnames是步骤1保存的 `model.keys()`，需要提前得到并替换进来，运行得到转换keys后的model. 这里以 YOLOv6m 转换为例：

```python
import json
import pickle
import numpy
import torch
import sys
import os
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# Run in original code
param_state_dict = torch.load('../weights/yolov6_v3/p5/yolov6m.pt')

old_state_dict = {}
param_state_dict = param_state_dict['model']
conv_bias = None
for k in param_state_dict.float().state_dict():
    if 'conv.bias' in k:
        conv_bias = param_state_dict.float().state_dict()[k].numpy()
    if 'conv.bias' not in k:
        if conv_bias is not None and 'bn.running_mean' in k:
            old_state_dict[k] = param_state_dict.float().state_dict()[k].numpy() - conv_bias
            conv_bias = None
        else:
            old_state_dict[k] = param_state_dict.float().state_dict()[k].numpy()

# load new model with the latest code and paste new model.state_dict().keys() 
newnames = ['backbone.stem.rbr_dense.conv.weight', 'backbone.stem.rbr_dense.bn.weight', ...]
print(f'num of newnames:{len(newnames)}')
print(f'num of old weights: {len(old_state_dict.keys())}')
for i in range(len(newnames)):
    print(newnames[i] + "   ******     " + list(old_state_dict.keys())[i])


tweights = {}
oldnames = list(old_state_dict.keys())
for i in range(len(newnames)):
    tweights[newnames[i]] = torch.tensor(old_state_dict[oldnames[i]])
ckpt = {'model': tweights}
torch.save(ckpt, 'weights/YOLOv6m_newtensor.pt')
```


### 3. 在4.0代码库下进行模型权重替换

```python
import torch
import sys
import os
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# Run in latest code
m = torch.load('runs/train/coco_yolov6m_distill_retry/weights/last_ckpt.pt') #步骤1保存的模型路径
state_dict = torch.load('../yolov6_3.0/weights/YOLOv6m_newtensor.pt') #步骤2转换权重键值后保存的模型路径
m['model'].load_state_dict(state_dict['model'])
m['ema'] = None
m['updates'] = None
m['optimizer'] = None
m['epoch'] = 299
torch.save(m, './weights/yolov6m.pt') # 保存最终转换后的模型文件
```
