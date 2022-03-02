import numpy as np
from utils.rotation import rotation_matrix_to_angle_axis
from utils.rot_6D import rot6D_to_angular, rot6d_to_rotmat_batch
from visualization.processing import transform_rot_representation

def convert_to3d(pred, targets, seq_len=None, type='9d'):
    seq_len_pred = pred.shape[1] if seq_len==None else seq_len
    pred_dict, gt_dict = {}, {}

    pred_pose9d = pred[:, :seq_len_pred, :216].reshape(-1, seq_len_pred, 24, 9).reshape(-1, seq_len_pred, 24, 3, 3)
    pred_pose9d = pred_pose9d.reshape(-1, 24, 3, 3).reshape(-1, 3, 3)

    pred_pose = rotation_matrix_to_angle_axis(pred_pose9d).reshape(-1, seq_len_pred, 24*3)
    pred_trans = pred[:, :seq_len_pred, 216:219]

    seq_len_tgt = targets.shape[1]
    targets_pose9d = targets[:, :seq_len_tgt, :216].reshape(-1, seq_len_tgt, 24, 9).reshape(-1, seq_len_tgt, 24, 3, 3)
    targets_pose9d = targets_pose9d.reshape(-1, 24, 3, 3).reshape(-1, 3, 3)

    targets_pose = rotation_matrix_to_angle_axis(targets_pose9d).reshape(-1, seq_len_tgt, 24*3)
    targets_trans = targets[:, :seq_len_tgt, 216:219]

    pred_dict['pose'], pred_dict['trans'] = pred_pose, pred_trans
    gt_dict['pose'], gt_dict['trans'] = targets_pose, targets_trans

    return pred_dict, gt_dict

def single_joint_convert_to3d(pred, seq_len=None, type='9d'):
    seq_len = pred.shape[1] if seq_len == None else seq_len
    pred_dict= {}

    pred_pose9d = pred[:, :seq_len, :216].reshape(-1, seq_len, 24, 9).reshape(-1, seq_len, 24, 3, 3)
    pred_pose9d = pred_pose9d.reshape(-1, 24, 3, 3).reshape(-1, 3, 3)

    pred_pose = rotation_matrix_to_angle_axis(pred_pose9d).reshape(-1, seq_len, 24 * 3)
    pred_trans = pred[:, :seq_len, 216:219]


    pred_dict['pose'], pred_dict['trans'] = pred_pose, pred_trans
    # gt_dict['pose'], gt_dict['trans'] = targets_pose, targets_trans

    return pred_dict