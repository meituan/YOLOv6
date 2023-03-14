python ./yolov6/utils/TestJson2VocClassTxt.py \
	--json_path " " \
	--save_path " "
python ./data/scripts/DOTA_devkit/ResultMerge_multi_process.py \
	--scrpath "" \
	--dstpath ""
# zip
cd dir
zip ../predictions.zip *
