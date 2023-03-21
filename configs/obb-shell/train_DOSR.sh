CUDA_VISIBLE_DEVICES=0 python tools/train_R.py \
	--device 0 \
	--batch 24 \
	--epochs 400 \
	--img 1024 \
	--eval-interval 20 \
	--conf configs/obb/yolov6s_finetune-obb.py \
	--data data/DOSR.yaml \
	--output-dir './runs/DOSR' \
	--name 'yolov6s_dfl_MGAR_AdamW'
# --write_trainbatch_tb
