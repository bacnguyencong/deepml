# activate python environment
source activate deepml-env

export CUDA_VISIBLE_DEVICES=1
logfile=$(date '+%d-%m-%Y-%H:%M:%S.txt');

DATA='Cub'
ARCH='bnincepnet'
LOSS='ContrastiveLoss'

LR=0.0001
DIM=128       # embedded feature size
IMG_SIZE=227  # image size
BATCH_SIZE=256  # batch size
EPOCHS=200      # number of epochs
CHECK_POINT='./output/model_best.pth.tar'

# run train python script
python train.py --data $DATA -a $ARCH -l $LOSS -img_size $IMG_SIZE -j 8 --lr $LR --epochs $EPOCHS --outdim $DIM -b $BATCH_SIZE --pretrained --seed 123456 > ./output/train.log

# run test python script
python test.py --data $DATA -a $ARCH -c $CHECK_POINT -img_size $IMG_SIZE -j 8 --outdim $DIM -b $BATCH_SIZE --seed 123456 > ./output/test.log