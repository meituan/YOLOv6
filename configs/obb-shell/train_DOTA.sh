CUDA_VISIBLE_DEVICES=0 python tools/train_R.py \
	--device 0 \
	--batch 16 \
	--epochs 80 \
	--img 800 \
	--eval-interval 20 \
	--conf configs/obb/yolov6l_finetjune-obb.py \
	--data data/DOTA.yaml \
	--output-dir './runs/DOTA' \
	--name 'yolov6l_dfl_MGAR_AdamW_small_lr'
# --resume './runs/drone_vehicle/yolov6s_dfl_csl/weights/last_ckpt.pt'
# --write_trainbatch_tb
