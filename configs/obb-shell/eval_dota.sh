epoch=35
eval_name="ss_l_1_$epoch"
eval_path="runs/DOTA-ss/test"
weight_path="./runs/DOTA-ss/yolov6l-obb/weights/${epoch}_ckpt.pt"
CUDA_VISIBLE_DEVICES=0 python tools/eval_R.py \
	--data "./data/DOTA-ss.yaml" \
	--weights $weight_path \
	--batch-size 64 \
	--img-size 1024 \
	--conf-thres 0.03 \
	--iou-thres 0.1 \
	--task "test" \
	--device 0 \
	--save_dir $eval_path \
	--name $eval_name \
	--test_load_size 1024 \
	--do_coco_metric False \
	--do_pr_metric False \
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

python yolov6/utils/TestJson2VocClassTxt.py \
	--json_path "$eval_path/$eval_name/predictions.json" \
	--save_path "$eval_path/$eval_name/predictions"

# merge
python data/scrpits/DOTA_devkit/ResultMerge_multi_process_new.py \
	--scrpath "$eval_path/$eval_name/predictions" \
	--dstpath "$eval_path/$eval_name/predictions_merge"

# zip
cd "$eval_path/$eval_name/predictions_merge"
zip ../predictions.zip *
