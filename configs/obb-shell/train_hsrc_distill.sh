CUDA_VISIBLE_DEVICES=0 python tools/train_R.py \
	--device 0 \
	--batch 32 \
	--epochs 300 \
	--img 800 \
	--eval-interval 20 \
	--conf configs/obb/yolov6n_finetune-obb.py \
	--data data/HRSC2016.yaml \
	--output-dir './runs/HRSC2016' \
	--name 'yolov6n_dfl_dfl' \
	--teacher_model_path 'runs/HRSC2016/exp6/weights/best_ckpt.pt' \
	--distill \
	--distill_feat \
	--temperature 20

# --write_trainbatch_tb
