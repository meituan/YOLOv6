python tools/train_R.py \
	--device 1 \
	--batch 32 \
	--epochs 400 \
	--img 800 \
	--eval-interval 20 \
	--conf configs/yolov6s_finetune-obb.py \
	--data data/HRSC2016.yaml \
	--output-dir './runs/HRSC2016' \
	--name 'exp' \
	# --write_trainbatch_tb
