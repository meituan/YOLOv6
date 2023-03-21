CUDA_VISIBLE_DEVICES=0 python tools/eval_R.py \
	--data "./data/drone_vehicle.yaml" \
	--weights "./runs/drone_vehicle/yolov6m_dfl_csl_AdamW_small_lr/weights/best_ckpt.pt" \
	--batch-size 32 \
	--img-size 800 \
	--conf-thres 0.1 \
	--iou-thres 0.65 \
	--task "val" \
	--device 0 \
	--save_dir "./runs/val/" \
	--name "exp" \
	--test_load_size 800 \
	--do_coco_metric False \
	--do_pr_metric True \
	--ap_method 'VOC12' \
	--verbose \
	--plot_confusion_matrix \
	--letterbox_return_int \
	--scale_exact
# --force_no_pad
# --not_infer_on_rect
# parser.add_argument('--reproduce_640_eval', default=False, action='store_true', help='whether to reproduce 640 infer result, overwrite some config')
# parser.add_argument('--eval_config_file', type=str, default='./configs/experiment/eval_640_repro.py', help='config file for repro 640 infer result')
# parser.add_argument('--config-file', default='', type=str, help='experiments description file, lower priority than reproduce_640_eval')
