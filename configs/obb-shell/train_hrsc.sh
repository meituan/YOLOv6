CUDA_VISIBLE_DEVICES=1 python tools/train_R.py \
	--device 0 \
	--batch 32 \
	--epochs 100 \
	--img 800 \
	--eval-interval 5 \
	--conf configs/obb/yolov6s_finetune-obb.py \
	--data data/HRSC2016.yaml \
	--output-dir './runs/HRSC2016' \
	--name 'yolov6n_dfl_dfl' \
	--bs_per_gpu 8
# --write_trainbatch_tb
