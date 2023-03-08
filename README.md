# 计划

## 3.7 任务 - YOLOv6 框架添加角度

> 分成五个部分进行分别改造，
>
>- 数据处理
>
>- 模型
> - 分配
> - loss
> - pr_metric

**角度部分维度变化比较多**，*维度可能得先考虑不要写死，写成活的，方便传参，和后面修改*

**传参的话使用网络cfg传递，某些函数可能没有传递cfg可能需要修改**

- 回归：1维
- CSL：180维
- DFL：90-180维
- MGAR：3-5，拆两分支，*这个先不考虑*

### 额外
- [ ] rich库, 更换LOGGING, 额外的LOGGING保存日志
### 数据处理

- [x] dataloader 部分label数据读取，添加angle维度
- [x] label 到 target 送入model和传到pr_metric中的维度变化
- [ ] letterbox bug
- [ ] 数据增强
### 模型

**重点关注train eval部分**

- [x] head头部分的定义，包含head_layers，和最终的detect部分，detect部分包含train和eval两种，需要仔细改动并加以考虑
- [ ] DFL 实现 check
- [ ] MGAR 实现
- [ ] MGAR 实现 check
- [ ] 额外的蒸馏和fuse_ab部分

### 分配

**分配部分主要关注TAL，这个部分考虑先仅改动hbb部分的维匹配，后面再引入角度**

- [ ] TAL部分考虑引入角度

### Loss

**Loss 计算部分有大量的xywh2xyxy变化和相对值到绝对值之间的转化，需要严查**

- [x] Angle loss的实现，包含回归，分类，dfl三种形式，先实现简单的
  - [x] 回归
  - [x] 分类
  - [x] dfl
  - [ ] MGAR

- [x] loss计算过程中的维度变化

### pr计算

**pr计算部分主要是两个部分，一个是nms()，一个是process_batch()**

- [x] nms部分有现成可以考虑，重点是维数变化，这个不能写死，**与v6版本做一下比较**
- [x] process_batch 部分也同样需要与v6版本做一下对比
- [ ] ap_pr_class 添加AP12,AP07实现
