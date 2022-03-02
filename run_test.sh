#NETWORK='Dancerevolution_2Transformer'
NETWORK='2tcnlstm'
VERSION='2'
EPOCH='1'
MODEL_DIR='/export2/home/lsy/music2dance_condition/checkpoints/'${NETWORK}'_V'${VERSION}
CUDA_VISIBLE_DEVICES=0 python core/test.py ${MODEL_DIR} ${NETWORK} ${EPOCH}
