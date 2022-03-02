#NETWORK='Dancerevolution_2Transformer'
#NETWORK='2tcnlstm'
NETWORK='CrossModalTr'
VERSION='5'
EPOCH='1500'
MODEL_DIR='/export/home/lg/huang/code/music2dance_condition/checkpoints/'${NETWORK}'_V'${VERSION}
CLASSIFIER_PATH="/export/home/lg/huang/code/music2dance_condition/classifier/best_model_512_512dim_60frame_rnn"
CUDA_VISIBLE_DEVICES=1 python core/test2.py ${MODEL_DIR} ${CLASSIFIER_PATH} ${NETWORK} ${EPOCH} ${VERSION} --regressive_len 60
