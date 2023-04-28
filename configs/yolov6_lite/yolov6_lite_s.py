# YOLOv6-lite-s model
model = dict(
    type='YOLOv6-lite-s',
    pretrained=None,
    width_multiple=0.7,
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
    lr0=0.1 * 4,
    lrf=0.01,
    momentum=0.9,
    weight_decay=0.00004,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1
)

data_aug = dict(
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
)
