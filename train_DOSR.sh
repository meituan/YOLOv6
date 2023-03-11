CUDA_VISIBLE_DEVICES=0 python tools/train_R.py \
	--device 0 \
	--batch 16 \
	--epochs 400 \
	--img 1024 \
	--eval-interval 20 \
	--conf configs/yolov6n_finetune-obb.py \
	--data data/DOSR.yaml \
	--output-dir './runs/DOSR' \
	--name 'yolov6n_dfl_MGAR'
# --write_trainbatch_tb
