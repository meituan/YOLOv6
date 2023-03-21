#
python yolov6/utils/TestJson2VocClassTxt.py \
	--json_path "./runs/DOTA/test/exp_65/predictions.json" \
	--save_path "./runs/DOTA/test/exp_65/predictions"

# merge
python data/scrpits/DOTA_devkit/ResultMerge_multi_process.py \
	--scrpath "./runs/DOTA/test/exp_65/predictions" \
	--dstpath "./runs/DOTA/test/exp_65/predictions_merge"

# zip
cd ./runs/DOTA/test/exp_65/predictions_merge
zip ../predictions.zip *
