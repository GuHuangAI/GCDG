#NETWORK='Dancerevolution_2Transformer'
NETWORK='2tcnlstm'
SAVE_VERSION='2'
SAVE_DIR='/export2/home/lsy/music2dance_condition/checkpoints/'${NETWORK}'_V'${SAVE_VERSION}
LOSS_VERSION='1'
CUDA_VISIBLE_DEVICES=0,1 python core/train.py ${SAVE_DIR} ${NETWORK} ${SAVE_VERSION} ${LOSS_VERSION}
