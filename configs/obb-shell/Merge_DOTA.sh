eval_name="yolov6n_352"
eval_path="runs/DOTA-test/test"

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
