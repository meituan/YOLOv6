python -m torch.distributed.launch --nproc_per_node 2 tools/train_R.py \
	--device 0,1 \
	--batch 16 \
	--epochs 36 \
	--img 1024 \
	--eval-interval 36 \
	--heavy-eval-range 0 \
	--conf configs/obb/yolov6l_finetune-obb.py \
	--data data/DOTA-ms.yaml \
	--output-dir './runs/DOTA-ms-baseline' \
	--name 'yolov6l_MGAR' \
	--save_ckpt_on_last_n_epoch 3 \
	--stop_aug_last_n_epoch 0 \
	--bs_per_gpu 16 \
	--write_trainbatch_tb
# --resume './runs/DOTA/yolov6l_dfl_MGAR_AdamW_small_lr_1024_1/weights/last_ckpt.pt'
