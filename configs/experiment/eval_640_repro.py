# eval param for different scale

eval_params = dict(
    default = dict(
        img_size=640,
        shrink_size=2,
        infer_on_rect=False,
    ),
    yolov6n = dict(
        img_size=640,
        shrink_size=4,
        infer_on_rect=False,
    ),
    yolov6t = dict(
        img_size=640,
        shrink_size=6,
        infer_on_rect=False,
    ),
    yolov6s = dict(
        img_size=640,
        shrink_size=6,
        infer_on_rect=False,
    ),
    yolov6m = dict(
        img_size=640,
        shrink_size=4,
        infer_on_rect=False,
    ),
    yolov6l = dict(
        img_size=640,
        shrink_size=4,
        infer_on_rect=False,
    ),
    yolov6l_relu = dict(
        img_size=640,
        shrink_size=2,
        infer_on_rect=False,
    ),
    yolov6n6 = dict(
        img_size=1280,
        shrink_size=17,
        infer_on_rect=False,
    ),
    yolov6s6 = dict(
        img_size=1280,
        shrink_size=8,
        infer_on_rect=False,
    ),
    yolov6m6 = dict(
        img_size=1280,
        shrink_size=64,
        infer_on_rect=False,
    ),
    yolov6l6 = dict(
        img_size=1280,
        shrink_size=41,
        infer_on_rect=False,
    ),
    yolov6s_mbla = dict(
        img_size=640,
        shrink_size=7,
        infer_on_rect=False,
    ),
    yolov6m_mbla = dict(
        img_size=640,
        shrink_size=7,
        infer_on_rect=False,
    ),
    yolov6l_mbla = dict(
        img_size=640,
        shrink_size=7,
        infer_on_rect=False,
    ),
    yolov6x_mbla = dict(
        img_size=640,
        shrink_size=3,
        infer_on_rect=False,
    )
)
