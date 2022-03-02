import os
import random
import pickle
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils import data
import _init_paths
from datasets import loader
from datasets import dataset_split_setting
from utils.rot9D_to3 import convert_to3d
from visualization import processing
from visualization import plot_motion_joints
from visualization import plot_music
from utils import fk
from datasets.dataset_split_setting import get_split
from configs.configs import video_list, motion_dir , music_dir , ignore_list, sequence_list, MF_CACHE_DIR,\
                    train_batch_size

SR = 60*512   # 30720

class AISTDataset_plot(object):
    """A dataset class for loading, processing and plotting AIST++"""

    def __init__(self, motion_list, music_list, motion_dir, music_dir):

        self.motion_list = motion_list
        self.music_list = music_list
        self.motion_dir = motion_dir
        self.music_dir = music_dir
        self.video_names = [l.strip() for l in self.motion_list]

        with open(ignore_list, 'r') as f:
            ignore_video_names = [l.strip() for l in f.readlines()] # test 18
            self.video_names = [name for name in self.video_names if name not in ignore_video_names] # 965

        with open(sequence_list, 'r') as f:
            all_sequence_names = [l.strip().split('\t')[0] for l in f.readlines()] # 13885
  
        self.sequence_names = [name for name in all_sequence_names if name[:25] in self.video_names] # train 10205; val 240; test 1110
        self.sequence_names= sorted(self.sequence_names)   
        
        self.fk_engine = fk.SMPLForwardKinematics()

    def __len__(self):
        return len(self.sequence_names)

    def __getitem__(self, index):
        sequence_name = self.sequence_names[index]
        return self.get_item(sequence_name)

    def get_item(self, video_name, verbose=True):
        seq_name, view = loader.get_seq_name(video_name[:25])
        frame = int(video_name[:-4].split('_')[-1])*60

        """motion"""
        choose_motion = {}
        motion_path = os.path.join(self.motion_dir, f'{seq_name}.pkl')
        motion_data = loader.load_pkl_lsy(motion_path, keys=['smpl_poses', 'smpl_scaling', 'smpl_trans'])
        motion_data['smpl_trans'] /= motion_data['smpl_scaling'] 
        del motion_data['smpl_scaling']
        del motion_data['smpl_loss']

        joints_pos = self.fk_engine.from_aa(motion_data['smpl_poses'], motion_data['smpl_trans'])

        smpl_joints = joints_pos[frame:frame+240]

        '''归一化'''
        mean1 = np.mean(smpl_joints)
        std1 = np.std(smpl_joints)
        smpl_joints = (smpl_joints- mean1)/std1 + np.finfo(float).eps

        motion_joints_current = smpl_joints[:120, ].reshape(-1, 72)
        motion_joints_future = smpl_joints[120:240, ].reshape(-1, 72)

        """audio"""
        with open(os.path.join(MF_CACHE_DIR, video_name[:-4]+'.pkl'), 'rb') as f:
            audio_features = pickle.load(f)

        '''归一化'''
        mean2 = np.mean(audio_features)
        std2 = np.std(audio_features)
        audio_features = (audio_features - mean2)/std2 + np.finfo(float).eps
        
        return audio_features, motion_joints_current, motion_joints_future


    @classmethod
    def plot_joints(cls, joints, save_path=None, save_type=None, fname=None, test=False, test_dir=None, epoch=None):
        """Visualize 3D joints."""
        out_dir = None
        if save_path:
            out_dir = os.path.dirname(save_path)
        fig = plot_motion_joints.animate_matplotlib(
            positions=[joints],
            colors=['r'],
            titles=[''],
            color_after_change='g',
            fig_title='smpl_joints',
            fps=60,
            figsize=(5.0, 5.0),
            out_dir=out_dir,
            save_type=save_type,
            fname=fname,
            test=test, 
            test_dir=test_dir,
            epoch=epoch
        )
        return fig
    
    @classmethod
    def plot_smpl_poses(cls, smpl_joints, save_path=None, save_type=None, fname=None,test=False, test_dir=None, epoch=None):
        fig = cls.plot_joints(smpl_joints.reshape(-1,24,3), save_path, save_type, fname, test=test, test_dir=test_dir, epoch=epoch)

        return fig