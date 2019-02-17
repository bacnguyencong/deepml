# activate python environment
source activate deepml-env

export CUDA_VISIBLE_DEVICES=1

DATA='Cub'
ARCH='bninception'
LOSS='Jeffrey'

LR=0.0001
DIM=128         # embedded feature size
IMG_SIZE=224    # image size
BATCH_SIZE=128  # batch size
EPOCHS=200      # number of epochs
CHECK_POINT='./output/model_best.pth.tar'
PRET='imagenet'

# run train python script
python train.py --data $DATA -a $ARCH -l $LOSS -img_size $IMG_SIZE -j 8 --lr $LR --epochs $EPOCHS --outdim $DIM -b $BATCH_SIZE --pretrained $PRET --seed 123456 

# --normalized

# run test python script
python test.py --data $DATA -a $ARCH -c $CHECK_POINT -img_size $IMG_SIZE -j 8 --outdim $DIM -b $BATCH_SIZE --pretrained $PRET --seed 123456

#--normalized