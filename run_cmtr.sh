NETWORK='CrossModalTr'
#NETWORK='2tcnlstm'
SAVE_VERSION='6_2'
SAVE_DIR='/export/home/lg/huang/code/music2dance_condition/checkpoints/'${NETWORK}'_V'${SAVE_VERSION}
LOSS_VERSION='2'
CUDA_VISIBLE_DEVICES=0,1,2,3 python core/train.py ${SAVE_DIR} ${NETWORK} ${SAVE_VERSION} ${LOSS_VERSION}