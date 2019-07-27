# activate python environment
source activate deepml-env

export CUDA_VISIBLE_DEVICES=2

DATA='Car'
ARCH='bninception'
LOSS='TripletLoss' #'SSTLoss'

LR=0.0001
DIM=512         # embedded feature size
IMG_SIZE=224    # image size
BATCH_SIZE=200  # batch size
EPOCHS=150      # number of epochs
CHECK_POINT='./output/model_best.pth.tar'
PRET='imagenet'
TAR=5

# run train python script
python train.py --data $DATA -a $ARCH -l $LOSS -n_targets $TAR -img_size $IMG_SIZE -j 4 --lr $LR --epochs $EPOCHS --outdim $DIM -b $BATCH_SIZE --pretrained $PRET

# run test python script
python test.py --data $DATA -a $ARCH -c $CHECK_POINT -img_size $IMG_SIZE -j 4 --outdim $DIM -b $BATCH_SIZE --pretrained $PRET