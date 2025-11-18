# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6

python src/runs/run_vlm_exp_ov.py \
--cfg cfg/vlm_exp_ov.yaml \
--gpu_ids 0,1,2,3,4,5,6,7