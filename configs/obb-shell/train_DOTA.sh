CUDA_VISIBLE_DEVICES=1 python tools/train_R.py \
	--device 0 \
	--batch 8 \
	--epochs 36 \
	--img 1024 \
	--eval-interval 3 \
	--conf configs/obb/yolov6s_finetune-obb.py \
	--data data/DOTA-ss.yaml \
	--output-dir "./runs/DOTA-ss" \
	--name "yolov6s_test" \
	--save_ckpt_on_last_n_epoch 5 \
	--stop_aug_last_n_epoch 0 \
	--bs_per_gpu 8 \
	--write_trainbatch_tb
# --resume './runs/drone_vehicle/yolov6s_dfl_csl/weights/last_ckpt.pt'
# --write_trainbatch_tb
