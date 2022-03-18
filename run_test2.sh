#NETWORK='Dancerevolution_2Transformer'
#NETWORK='2tcnlstm'
NETWORK='CrossModalTr'
VERSION='13'
EPOCH='1600'
MODEL_DIR='./checkpoints/'${NETWORK}'_V'${VERSION}'_3'
CLASSIFIER_PATH="./classifier/best_model_512_512dim_100frame_rnn_smpl"
CUDA_VISIBLE_DEVICES=1 python core/test2.py ${MODEL_DIR} ${CLASSIFIER_PATH} ${NETWORK} ${EPOCH} ${VERSION} --regressive_len 60
