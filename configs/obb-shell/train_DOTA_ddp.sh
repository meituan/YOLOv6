python -m torch.distributed.launch --nproc_per_node 2 tools/train_R.py \
	--device 0,1 \
	--batch 32 \
	--epochs 36 \
	--img 1024 \
	--eval-interval 3 \
	--conf configs/obb/yolov6s_finetune-obb.py \
	--data data/DOTA-ss.yaml \
	--output-dir './runs/DOTA' \
	--name 'yolov6l_Combine_obb_36_regression' \
	--save_ckpt_on_last_n_epoch 5 \
	--stop_aug_last_n_epoch 0 \
	--bs_per_gpu 16 \
	--write_trainbatch_tb
# --resume './runs/DOTA/yolov6l_dfl_MGAR_AdamW_small_lr_1024_1/weights/last_ckpt.pt'
