python tools/train_R.py \
	--device 0 \
	--batch 32 \
	--img 800 \
	--eval-interval 20 \
	--conf configs/yolov6s_finetune-obb.py \
	--data data/drone_vehicle.yaml \
	--output-dir './runs/drone_vehicle' \
	--name 'exp' \
	--write_trainbatch_tb
