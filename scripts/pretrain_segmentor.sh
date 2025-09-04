GPU_ID=1

#DATASET='s3dis'
#SPLIT=0
#DATA_PATH='./datasets/S3DIS/scenes/blocks_bs1_s1'
#SAVE_PATH='./log_s3dis/'

# DATASET='s3dis_model'
# SPLIT=0
# DATA_PATH='./datasets/S3DIS/scenes/blocks_bs1_s1'
# SAVE_PATH='./log_s3dis_modelnet/'

DATASET='s3dis_sunrgbd'
SPLIT=0
DATA_PATH='./datasets/S3DIS/scenes/blocks_bs1_s1'
SAVE_PATH='./log_s3dis_sunrgbd/'

# DATASET='scannet'
# SPLIT=1
# DATA_PATH='./datasets/ScanNet/blocks_bs1_s1'
# SAVE_PATH='./log_scannet/'

# DATASET='scannet_model'
# SPLIT=1
# DATA_PATH='./datasets/ScanNet/blocks_bs1_s1'
# SAVE_PATH='./log_scannet_modelnet/'

# DATASET='scannet_sunrgbd'
# SPLIT=1
# DATA_PATH='./datasets/ScanNet/blocks_bs1_s1'
# SAVE_PATH='./log_scannet_sunrgbd/'

NUM_POINTS=2048
PC_ATTRIBS='xyzrgbXYZ'
EDGECONV_WIDTHS='[[64,64], [64, 64], [64, 64]]'
MLP_WIDTHS='[512, 256]'
K=20

EVAL_INTERVAL=3
BATCH_SIZE=16
NUM_WORKERS=16
NUM_EPOCHS=150
LR=0.001
WEIGHT_DECAY=0.0001
DECAY_STEP=50
DECAY_RATIO=0.5

args=(--phase 'pretrain' --dataset "${DATASET}" --cvfold $SPLIT
      --data_path  "$DATA_PATH" --save_path "$SAVE_PATH"
      --pc_npts $NUM_POINTS --pc_attribs "$PC_ATTRIBS" --pc_augm
      --edgeconv_widths "$EDGECONV_WIDTHS" --dgcnn_k $K
      --dgcnn_mlp_widths "$MLP_WIDTHS" --use_high_dgcnn --pc_augm_scale 1.25 --pc_augm_shift 0.1
      --n_iters $NUM_EPOCHS --eval_interval $EVAL_INTERVAL
      --batch_size $BATCH_SIZE --n_workers $NUM_WORKERS
      --pretrain_lr $LR --pretrain_weight_decay $WEIGHT_DECAY
      --pretrain_step_size $DECAY_STEP --pretrain_gamma $DECAY_RATIO)

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}"
