CUDA_VISIBLE_DEVICES=1 python tools/train_R.py \
	--device 0 \
	--batch 16 \
	--img 800 \
	--eval-interval 20 \
	--conf configs/yolov6m_finetune-obb-drone.py \
	--data data/drone_vehicle.yaml \
	--output-dir './runs/drone_vehicle' \
	--name 'yolov6m_dfl_csl'
# --write_trainbatch_tb
