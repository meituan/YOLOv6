# YOLOv6n model
model = dict(
    type='YOLOv6n6',
    pretrained=None,
    depth_multiple=0.33,
    width_multiple=0.25,
    backbone=dict(
        type='EfficientRep6',
        num_repeats=[1, 6, 12, 18, 6, 6],
        out_channels=[64, 128, 256, 512, 768, 1024],
        fuse_P2=True, # if use RepBiFPANNeck6, please set fuse_P2 to True.
        cspsppf=True,
        ),
    neck=dict(
        type='RepBiFPANNeck6',
        num_repeats=[12, 12, 12, 12, 12, 12],
        out_channels=[512, 256, 128, 256, 512, 1024],
        ),
    head=dict(
        type='EffiDeHead',
        in_channels=[128, 256, 512, 1024],
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
    lr0=0.02,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
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
