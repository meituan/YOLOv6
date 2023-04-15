CUDA_VISIBLE_DEVICES=0 python tools/train_R.py \
	--device 0 \
	--batch 8 \
	--epochs 36 \
	--img 1024 \
	--eval-interval 12 \
	--heavy-eval-range 0 \
	--conf configs/obb/yolov6s_finetune-obb.py \
	--data data/DOTA_ss_new.yaml \
	--output-dir "./runs/DOTA-ss-baseline" \
	--name "yolov6s-MGAR-ss-new-aug" \
	--save_ckpt_on_last_n_epoch 3 \
	--stop_aug_last_n_epoch 0 \
	--bs_per_gpu 8
# --write_trainbatch_tb
# --resume './runs/drone_vehicle/yolov6s_dfl_csl/weights/last_ckpt.pt'
