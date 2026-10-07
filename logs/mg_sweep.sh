set -e
export TRITON_CACHE_DIR=/tmp/tc_mg_$SLURM_LOCALID
B=1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768
HUGE=1073741824
# Megatron-shaped (fp32 expert-output buffer): the three strategies, one per run.
# 1) AGv dispatch + RSv combine  = what 'nvls' does
torchrun --nproc_per_node=4 --master_port=29520 run.py --impl dynamic --dyn-out-dtype fp32 \
  --dyn-dispatch-threshold $HUGE --dyn-combine-threshold $HUGE \
  --batch-sizes $B --num-layers 88 --reps 30 --warmup 8 --out mg_agv_rsv.csv
# 2) A2Av dispatch + RSv combine = the middle hybrid (isolates the DISPATCH crossover)
torchrun --nproc_per_node=4 --master_port=29520 run.py --impl dynamic --dyn-out-dtype fp32 \
  --dyn-dispatch-threshold 0 --dyn-combine-threshold $HUGE \
  --batch-sizes $B --num-layers 88 --reps 30 --warmup 8 --out mg_a2av_rsv.csv
# 3) A2Av dispatch + push combine = what 'a2av' does (isolates the COMBINE crossover)
torchrun --nproc_per_node=4 --master_port=29520 run.py --impl dynamic --dyn-out-dtype fp32 \
  --dyn-dispatch-threshold 0 --dyn-combine-threshold 0 \
  --batch-sizes $B --num-layers 88 --reps 30 --warmup 8 --out mg_a2av_push.csv
echo ALL_DONE
