# YOLOv6-lite-l model
model = dict(
    type='YOLOv6-lite-l',
    pretrained='weights/yolov6lite_l.pt',
    width_multiple=1.5,
    backbone=dict(
        type='Lite_EffiBackbone',
        num_repeats=[1, 3, 7, 3],
        out_channels=[24, 32, 64, 128, 256],
        scale_size=0.5,
        ),
    neck=dict(
        type='Lite_EffiNeck',
        in_channels=[256, 128, 64],
        unified_channels=96
        ),
    head=dict(
        type='Lite_EffideHead',
        in_channels=[96, 96, 96, 96],
        num_layers=4,
        anchors=1,
        strides=[8, 16, 32, 64],
        atss_warmup_epoch=4,
        iou_type='siou',
        use_dfl=False,
        reg_max=0 #if use_dfl is False, please set reg_max to 0
    )
)

solver = dict(
    optim='SGD',
    lr_scheduler='Cosine',
    lr0=0.0032,
    lrf=0.12,
    momentum=0.843,
    weight_decay=0.00036,
    warmup_epochs=2.0,
    warmup_momentum=0.5,
    warmup_bias_lr=0.05
)

data_aug = dict(
    hsv_h=0.0138,
    hsv_s=0.664,
    hsv_v=0.464,
    degrees=0.373,
    translate=0.245,
    scale=0.898,
    shear=0.602,
    flipud=0.00856,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.243,
)
