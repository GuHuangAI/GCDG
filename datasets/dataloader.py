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
from visualization import plot_motion
from visualization import plot_music
from utils import fk
from datasets.dataset_split_setting import get_split
# from configs.configs_cmtr import config video_list, motion_dir , music_dir , ignore_list, sequence_list, MF_CACHE_DIR,\
#                     train_batch_size

SR = 60*512
cache_path ='./motion.pkl'

# for Transformer
def paired_collate_fn1(insts):
    audio_seq, tgt_seq, video_name  = list(zip(*insts))
    tgt_seq = torch.FloatTensor(tgt_seq)

    condition_motion = tgt_seq[:, :120, :]
    tgt_motion = tgt_seq[:, 120:, :]
    condition_motion = torch.FloatTensor(condition_motion)
    tgt_motion = torch.FloatTensor(tgt_motion)

    src_pos1 = np.array([[pos_i + 1 for pos_i, v_i in enumerate(inst)] for inst in audio_seq])
    src_pos2 = np.array([[pos_i + 1 for pos_i, v_i in enumerate(inst)] for inst in condition_motion])

    audio_seq = torch.FloatTensor(audio_seq)
    src_pos1 = torch.LongTensor(src_pos1)
    src_pos2 = torch.LongTensor(src_pos2)

    return (audio_seq, src_pos1), (condition_motion, src_pos2), tgt_motion, video_name

# for TCN
def paired_collate_fn2(insts):
    audio_seq, tgt_seq, video_name = list(zip(*insts))
    tgt_seq = torch.FloatTensor(tgt_seq)

    condition_motion = tgt_seq[:, :120, :]
    tgt_motion = tgt_seq[:, 120:, :]

    condition_motion = torch.FloatTensor(condition_motion)
    tgt_motion = torch.FloatTensor(tgt_motion)
    audio_seq = torch.FloatTensor(audio_seq)

    return audio_seq, condition_motion, tgt_motion, video_name

def paired_collate_fn3(insts):
    audio_seq, tgt_seq, video_name  = list(zip(*insts))
    tgt_seq = torch.FloatTensor(tgt_seq)

    condition_motion = tgt_seq[:, :120, :]
    tgt_motion = tgt_seq[:, 120:180, :]
    condition_motion = torch.FloatTensor(condition_motion)
    tgt_motion = torch.FloatTensor(tgt_motion)

    src_pos1 = np.array([[pos_i + 1 for pos_i, v_i in enumerate(inst)] for inst in audio_seq])
    src_pos2 = np.array([[pos_i + 1 for pos_i, v_i in enumerate(inst)] for inst in condition_motion])

    audio_seq = torch.FloatTensor(audio_seq)
    src_pos1 = torch.LongTensor(src_pos1)
    src_pos2 = torch.LongTensor(src_pos2)

    return (audio_seq, src_pos1), (condition_motion, src_pos2), tgt_motion, video_name

class AISTDataset(object):
    """A dataset class for loading, processing and plotting AIST++"""
    def __init__(self, motion_list, music_list, motion_dir, music_dir, config, sample=-1):
        self.motion_list = motion_list
        self.music_list = music_list
        self.motion_dir = motion_dir
        self.music_dir = music_dir
        self.config = config
        self.video_names = [l.strip() for l in self.motion_list]
        
        with open(config.ignore_list, 'r') as f:
            ignore_video_names = [l.strip() for l in f.readlines()] # test 18
            self.video_names = [name for name in self.video_names if name not in ignore_video_names] # 965

        with open(config.sequence_list, 'r') as f:
            all_sequence_names = [l.strip().split('\t')[0] for l in f.readlines()] # 13885
  
        self.sequence_names = [name for name in all_sequence_names if name[:25] in self.video_names] # train 10205; val 240; test 1110
        self.sequence_names= sorted(self.sequence_names)
        if sample > 0:
            self.sequence_names = self.sequence_names[:sample]
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

        for k, v in motion_data.items(): 
            choose_motion[k] = motion_data[k][frame:frame+240]


        smpl_poses = np.zeros((240, 24, 9)) 
        choose_motion['smpl_poses'] = choose_motion['smpl_poses'].reshape(-1, 24, 3)

        for i in range(240):
            smpl_poses[i, ] = np.apply_along_axis(processing.transform_rot_representation, 1, choose_motion['smpl_poses'][i, ]).reshape(24, 9)
        smpl_poses = smpl_poses.reshape(-1, 216)

        final_motion = np.column_stack((smpl_poses, choose_motion['smpl_trans']))

        """audio"""
        with open(os.path.join(self.config.MF_CACHE_DIR, video_name[:-4]+'.pkl'), 'rb') as f:
            audio_features = pickle.load(f)
        
        audio_features = audio_features[:240, :]

        return audio_features, final_motion, video_name[:-4]

    @classmethod
    def plot_joints(cls, joints, save_path=None, save_type=None, fname=None, \
                    test=False, test_dir=None, epoch=None, to_video=False, audio_path=None):
        """Visualize 3D joints."""
        out_dir = None
        if save_path:
            out_dir = os.path.dirname(save_path)
            # fname = os.path.basename(save_path).split('.')[0]
        fig = plot_motion.animate_matplotlib(
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
            epoch=epoch,
            to_video=to_video, 
            audio_path=audio_path
        )
        return fig
    
    @classmethod
    def plot_smpl_poses(cls, smpl_poses, smpl_trans=None, save_path=None, save_type=None, \
                        fname=None,test=False, test_dir=None, epoch=None, to_video=False, audio_path=None):
        """Visualize SMPL joint angles, along with global translation."""
        fk_engine = fk.SMPLForwardKinematics()
        joints = fk_engine.from_aa(smpl_poses, smpl_trans)
        fig = cls.plot_joints(joints, save_path, save_type, fname, test=test, \
                test_dir=test_dir, epoch=epoch, to_video=to_video, audio_path=audio_path)

        return fig
