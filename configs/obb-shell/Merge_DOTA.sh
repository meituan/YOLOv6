eval_name="exp_78_1"
eval_path="runs/DOTA/test"

python yolov6/utils/TestJson2VocClassTxt.py \
	--json_path "$eval_path/$eval_name/predictions.json" \
	--save_path "$eval_path/$eval_name/predictions"

# merge
python data/scrpits/DOTA_devkit/ResultMerge_multi_process.py \
	--scrpath "$eval_path/$eval_name/predictions" \
	--dstpath "$eval_path/$eval_name/predictions_merge"

# zip
cd "$eval_path/$eval_name/predictions_merge"
zip ../predictions.zip *
