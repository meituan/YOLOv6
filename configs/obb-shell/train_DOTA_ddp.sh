python -m torch.distributed.launch --nproc_per_node 2 tools/train_R.py \
	--device 0,1 \
	--batch 32 \
	--epochs 80 \
	--img 800 \
	--eval-interval 20 \
	--conf configs/obb/yolov6l_finetune-obb.py \
	--data data/DOTA.yaml \
	--output-dir './runs/DOTA' \
	--name 'yolov6l_dfl_MGAR_AdamW_small_lr' \
	--save_ckpt_on_last_n_epoch 20
# --resume './runs/drone_vehicle/yolov6s_dfl_csl/weights/last_ckpt.pt'
# --write_trainbatch_tb
