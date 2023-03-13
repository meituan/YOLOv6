# 计划

## 3.7 任务 - YOLOv6 框架添加角度

> 分成五个部分进行分别改造，
>
> - 数据处理
>
> - 模型
> - 分配
> - loss
> - pr_metric

**角度部分维度变化比较多**，_维度可能得先考虑不要写死，写成活的，方便传参，和后面修改_

**传参的话使用网络 cfg 传递，某些函数可能没有传递 cfg 可能需要修改**

- 回归：1 维
- CSL：180 维
- DFL：90-180 维
- MGAR：3-5，拆两分支，_这个先不考虑_

### 额外

- [x] rich 库, 更换 LOGGING
- [ ] LOGGING 保存日志, 存储到 txt 中, 这块没法指定,有点烦人

### 数据处理

- [x] dataloader 部分 label 数据读取，添加 angle 维度
- [x] label 到 target 送入 model 和传到 pr_metric 中的维度变化
- [x] letterbox bug
- [x] 数据增强

### 模型

**重点关注 train eval 部分**

- [x] head 头部分的定义，包含 head_layers，和最终的 detect 部分，detect 部分包含 train 和 eval 两种，需要仔细改动并加以考虑
- [x] DFL 实现 check
- [x] MGAR 实现
- [x] MGAR 实现 check
- [ ] 额外的蒸馏和 fuse_ab 部分

### 分配

**分配部分主要关注 TAL，这个部分考虑先仅改动 hbb 部分的维匹配，后面再引入角度**

- [ ] TAL 部分考虑引入角度

### Loss

**Loss 计算部分有大量的 xywh2xyxy 变化和相对值到绝对值之间的转化，需要严查**

- [x] Angle loss 的实现，包含回归，分类，dfl 三种形式，先实现简单的

  - [x] 回归
  - [x] 分类
  - [x] dfl
  - [x] MGAR

- [x] loss 计算过程中的维度变化

### pr 计算

**pr 计算部分主要是两个部分，一个是 nms()，一个是 process_batch()**

- [x] nms 部分有现成可以考虑，重点是维数变化，这个不能写死，**与 v6 版本做一下比较**
- [x] process_batch 部分也同样需要与 v6 版本做一下对比
- [x] ap_pr_class 添加 AP12,AP07 实现
