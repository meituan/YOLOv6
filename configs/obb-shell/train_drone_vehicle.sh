CUDA_VISIBLE_DEVICES=1 python tools/train_R.py \
	--device 0 \
	--batch 16 \
	--epochs 100 \
	--img 800 \
	--eval-interval 20 \
	--conf configs/obb/yolov6l_finetune-obb-drone.py \
	--data data/drone_vehicle.yaml \
	--output-dir './runs/drone_vehicle' \
	--name 'yolov6l_dfl_csl_AdamW_small_lr'
# --resume './runs/drone_vehicle/yolov6s_dfl_csl/weights/last_ckpt.pt'
# --write_trainbatch_tb
