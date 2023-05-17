# YOLOv6l6 model
model = dict(
    type='YOLOv6l6',
    pretrained='weights/yolov6l6.pt',
    depth_multiple=1.0,
    width_multiple=1.0,
    backbone=dict(
        type='CSPBepBackbone_P6',
        num_repeats=[1, 6, 12, 18, 6, 6],
        out_channels=[64, 128, 256, 512, 768, 1024],
        csp_e=float(1)/2,
        fuse_P2=True,
        ),
    neck=dict(
        type='CSPRepBiFPANNeck_P6',
        num_repeats=[12, 12, 12, 12, 12, 12],
        out_channels=[512, 256, 128, 256, 512, 1024],
        csp_e=float(1)/2,
        ),
    head=dict(
        type='EffiDeHead',
        in_channels=[128, 256, 512, 1024],
        num_layers=4,
        anchors=1,
        strides=[8, 16, 32, 64],
        atss_warmup_epoch=4,
        iou_type='giou',
        use_dfl=True,
        reg_max=16, #if use_dfl is False, please set reg_max to 0
        distill_weight={
            'class': 1.0,
            'dfl': 1.0,
        },
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
training_mode = "conv_silu"
