import pickle
import os
import cv2
import sys
sys.path.append('..')
from datasets.dataset_split_setting import get_split
from datasets import loader
from configs.configs import config
video_list = config.video_list
motion_dir = config.motion_dir
MF_CACHE_DIR = config.MF_CACHE_DIR
video_list = r'G:\database\TTA_data\video_list.txt'
with open(video_list, 'r') as f:
    video_names = [l.strip() for l in f.readlines()]

val_split = get_split(video_names, task='generation', subset='val', is_paired=True)
val_video_names = val_split['video_names']
test_split = get_split(video_names, task='generation', subset='test', is_paired=False)
test_video_names = test_split['video_names']
test_split2 = get_split(video_names, task='generation', subset='test', is_paired=True)
test_video_names2 = test_split2['video_names']

train_split = get_split(video_names, task='prediction', subset='train')
train_video_names = train_split['video_names']
# print(len(test_video_names))
# test_music_names = test_split['music_names']
if __name__ == '__main__':
    # import numpy as np
    # path = r'G:\0001.jpg'
    # img = cv2.imread(path, 1)
    # imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
    # imgrgb = imgrgb.astype(np.float32)
    # # out = cv2.ximgproc_StructuredEdgeDetection.detectEdges(imgrgb)
    # model = 'structured edge/model.yml'
    # retval = cv2.ximgproc.createStructuredEdgeDetection(model)
    # out = retval.detectEdges(imgrgb)


    print('1111')
    # for seq_name in test_video_names:
    #     name, view = loader.get_seq_name(seq_name[:25])
    #     with open(os.path.join(config.test_cache_dir, seq_name + '.pkl'), 'rb') as f:
    #         audio_features = pickle.load(f)
    #     motion_path = os.path.join(motion_dir, f'{name}.pkl')
    #     motion_data = loader.load_pkl_lsy(motion_path, keys=['smpl_poses', 'smpl_scaling', 'smpl_trans'])
    #     motion_data['smpl_trans'] /= motion_data['smpl_scaling']
    #     #if motion_data['smpl_trans'].shape[0]>1200:
    #     print(name)
    #     print(motion_data['smpl_trans'].shape)
    #     print(audio_features.shape)
    # for seq_name in video_names:
    #     # name, view = loader.get_seq_name(seq_name[:25])
    #     with open(os.path.join(config.test_cache_dir, seq_name + '.pkl'), 'rb') as f:
    #         audio_features = pickle.load(f)
    #     txt_path = "/export2/home/lsy/TTA_data/longseq.txt"
    #     if audio_features.shape[0]>1350:
    #         with open(txt_path, 'a+') as f:
    #             f.write(seq_name)
    #             f.write('\n')
    ignore_list = r'G:\database\TTA_data\ignore_list.txt'
    long_list = r'G:\database\TTA_data\longseq.txt'
    with open(ignore_list, 'r') as f:
        ignore_list_names = [l.strip() for l in f.readlines()]
    with open(long_list, 'r') as f:
        long_list_names = [l.strip() for l in f.readlines()]
    tmp_list = [name for name in long_list_names if name not in ignore_list_names and name not in train_video_names]
    tmp_list2 = [name for name in long_list_names if name not in train_video_names]
    cv2.ximgproc_StructuredEdgeDetection
    pass

