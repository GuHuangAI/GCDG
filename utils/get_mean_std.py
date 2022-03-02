import os
import tqdm
import pickle
import numpy as np
from datasets import loader
from visualization import processing
from configs.configs import _DIR, video_list, motion_dir, ignore_list


def get_mean_std(train_video_names, motion_dir, if_notfirst=True):
    if if_notfirst:

        with open('./mean.pkl', 'rb') as f1:
            mean = pickle.load(f1)

        with open('./std.pkl', 'rb') as f2:
            std = pickle.load(f2)

        return mean, std

    else:
        video_names = [l.strip() for l in train_video_names]
        # print(len(video_names)) # 980

        with open(ignore_list, 'r') as f:
            ignore_video_names = [l.strip() for l in f.readlines()]
            video_names = [name for name in video_names if name not in ignore_video_names]
        # print(len(video_names)) # 965

        motion_list = []
        for name in tqdm.tqdm(video_names):
            splits = name.split('_')
            splits[2] = 'cAll'
            motion_file = '_'.join(splits)
            motion_path = os.path.join(motion_dir, motion_file + '.pkl')

            motion_data = loader.load_pkl_lsy(motion_path, keys=['smpl_poses', 'smpl_scaling', 'smpl_trans'])
            motion_data['smpl_trans'] /= motion_data['smpl_scaling'] 
            del motion_data['smpl_scaling']
            del motion_data['smpl_loss']

            """motion的3维旋转角变为旋转矩阵"""
            smpl_poses = np.zeros((motion_data['smpl_poses'].shape[0], 24, 9)) 
            motion_data['smpl_poses'] = motion_data['smpl_poses'].reshape(-1, 24, 3)
            for i in range(motion_data['smpl_poses'].shape[0]):
                smpl_poses[i, ] = np.apply_along_axis(processing.transform_rot_representation, 1, motion_data['smpl_poses'][i, ]).reshape(24, 9)
            smpl_poses = smpl_poses.reshape(-1, 216)

            motion = np.column_stack((smpl_poses, motion_data['smpl_trans']))
            motion_list.append(motion)
        
        motion_list = np.concatenate([v for v in motion_list], axis=0)

        # motion_array = np.mat(motion_list)
        print(motion_list.shape)

        mean = np.mean(motion_list, axis=(0,1))
        std = np.std(motion_list, axis=(0,1))

        with open('./mean.pkl', 'wb') as f:
            pickle.dump(mean, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open('./std.pkl', 'wb') as f:
            pickle.dump(std, f, protocol=pickle.HIGHEST_PROTOCOL)

        return mean, std