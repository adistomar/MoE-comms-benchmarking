set -e
export TRITON_CACHE_DIR=/tmp/tc_fine_$SLURM_LOCALID
B=1024,1152,1280,1408,1536,1664,1792,1920,2048
HUGE=1073741824
torchrun --nproc_per_node=4 --master_port=29540 run.py --impl dynamic --dyn-out-dtype fp32 \
  --dyn-dispatch-threshold $HUGE --dyn-combine-threshold $HUGE \
  --batch-sizes $B --num-layers 88 --reps 40 --warmup 10 --out fine_agv_rsv.csv
torchrun --nproc_per_node=4 --master_port=29540 run.py --impl dynamic --dyn-out-dtype fp32 \
  --dyn-dispatch-threshold 0 --dyn-combine-threshold 0 \
  --batch-sizes $B --num-layers 88 --reps 40 --warmup 10 --out fine_a2av_push.csv
torchrun --nproc_per_node=4 --master_port=29540 run.py --impl dynamic --dyn-out-dtype fp32 \
  --dyn-dispatch-threshold 0 --dyn-combine-threshold $HUGE \
  --batch-sizes $B --num-layers 88 --reps 40 --warmup 10 --out fine_a2av_rsv.csv
echo ALL_DONE
