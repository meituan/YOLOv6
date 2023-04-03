CUDA_VISIBLE_DEVICES=0 python tools/train_R.py \
	--device 0 \
	--batch 16 \
	--epochs 36 \
	--img 1024 \
	--eval-interval 12 \
	--conf configs/obb/yolov6n_finetune-obb.py \
	--data data/DOTA-ss.yaml \
	--output-dir "./runs/DOTA-ss-baseline" \
	--name "yolov6n-csl-nodetach" \
	--save_ckpt_on_last_n_epoch 3 \
	--stop_aug_last_n_epoch 0 \
	--bs_per_gpu 16
# --write_trainbatch_tb
# --resume './runs/drone_vehicle/yolov6s_dfl_csl/weights/last_ckpt.pt'
# --write_trainbatch_tbl
