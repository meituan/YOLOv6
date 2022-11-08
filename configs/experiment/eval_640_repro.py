# eval param for different scale

eval_params = dict(
    default = dict(
        img_size=640,
        test_load_size=638,
        letterbox_return_int=True,
        scale_exact=True,
        force_no_pad=True,
        not_infer_on_rect=True,
    ),
    yolov6n = dict(
        img_size=640,
        test_load_size=638,
        letterbox_return_int=True,
        scale_exact=True,
        force_no_pad=True,
        not_infer_on_rect=True,
    ),
    yolov6t = dict(
        img_size=640,
        test_load_size=634,
        letterbox_return_int=True,
        scale_exact=True,
        force_no_pad=True,
        not_infer_on_rect=True,
    ),
    yolov6s = dict(
        img_size=640,
        test_load_size=638,
        letterbox_return_int=True,
        scale_exact=True,
        force_no_pad=True,
        not_infer_on_rect=True,
    ),
    yolov6m = dict(
        img_size=640,
        test_load_size=628,
        letterbox_return_int=True,
        scale_exact=True,
        force_no_pad=True,
        not_infer_on_rect=True,
    ),
    yolov6l = dict(
        img_size=640,
        test_load_size=632,
        letterbox_return_int=True,
        scale_exact=True,
        force_no_pad=True,
        not_infer_on_rect=True,
    ),
    yolov6l_relu = dict(
        img_size=640,
        test_load_size=638,
        letterbox_return_int=True,
        scale_exact=True,
        force_no_pad=True,
        not_infer_on_rect=True,
    )
)
