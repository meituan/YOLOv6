# YOLOv6n model with eval param(when traing)
model = dict(
    type='YOLOv6n',
    pretrained=None,
    depth_multiple=0.33,
    width_multiple=0.25,
    backbone=dict(
        type='EfficientRep',
        num_repeats=[1, 6, 12, 18, 6],
        out_channels=[64, 128, 256, 512, 1024],
        ),
    neck=dict(
        type='RepPANNeck',
        num_repeats=[12, 12, 12, 12],
        out_channels=[256, 128, 128, 256, 256, 512],
        ),
    head=dict(
        type='EffiDeHead',
        in_channels=[128, 256, 512],
        num_layers=3,
        begin_indices=24,
        anchors=1,
        out_indices=[17, 20, 23],
        strides=[8, 16, 32],
        iou_type='siou',
        use_dfl=False,
        reg_max=0 #if use_dfl is False, please set reg_max to 0
    )
)

solver = dict(
    optim='SGD',
    lr_scheduler='Cosine',
    lr0=0.02, #0.01 # 0.02
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

# Eval params when eval model.
# If eval_params item is list, eg conf_thres=[0.03, 0.03],
# first will be used in train.py and second will be used in eval.py.
eval_params = dict(
    batch_size=None,  #None mean will be the same as batch on one device * 2
    img_size=None,  #None mean will be the same as train image size
    conf_thres=0.03,
    iou_thres=0.65,

    #pading and scale coord
    test_load_size=None, #None mean will be the same as test image size
    letterbox_return_int=False,
    force_no_pad=False,
    not_infer_on_rect=False,
    scale_exact=False,

    #metric
    verbose=False,
    do_coco_metric=True,
    do_pr_metric=False,
    plot_curve=False,
    plot_confusion_matrix=False
)
