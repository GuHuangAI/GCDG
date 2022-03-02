from collections import OrderedDict
import pickle
import os
import librosa
import numpy as np
import scipy
import scipy.signal as scisignal
import tqdm
import math
from dtw import dtw
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R


# ===========================================================
# Frechet Distance
# ===========================================================
def euc_dist(pt1, pt2):
    return math.sqrt((pt2[0]-pt1[0])*(pt2[0]-pt1[0])+(pt2[1]-pt1[1])*(pt2[1]-pt1[1]))
 
def _c(ca,i,j,P,Q):
    if ca[i,j] > -1:
        return ca[i,j]
    elif i == 0 and j == 0:
        ca[i,j] = euc_dist(P[0],Q[0])
    elif i > 0 and j == 0:
        ca[i,j] = max(_c(ca,i-1,0,P,Q),euc_dist(P[i],Q[0]))
    elif i == 0 and j > 0:
        ca[i,j] = max(_c(ca,0,j-1,P,Q),euc_dist(P[0],Q[j]))
    elif i > 0 and j > 0:
        ca[i,j] = max(min(_c(ca,i-1,j,P,Q),_c(ca,i-1,j-1,P,Q),_c(ca,i,j-1,P,Q)),euc_dist(P[i],Q[j]))
    else:
        ca[i,j] = float("inf")
    return ca[i,j]
 
def frechet_distance(P,Q):
    ca = np.ones((len(P),len(Q)))
    ca = np.multiply(ca,-1)
    return _c(ca, len(P) - 1, len(Q) - 1, P, Q)  # ca是a*b的矩阵(3*4),2,3


def pos_frechet(predict_joints, target_joints):
    pos_f = 0
    seq_len = predict_joints.shape[0]
    for i in range(seq_len):
        pos_f += frechet_distance(predict_joints[i, ], target_joints[i, ])
    return pos_f


def vel_frechet(predict_joints, target_joints):
    vel_f = 0
    velocity_predicted = np.diff(predict_joints, axis=0)
    velocity_target = np.diff(target_joints, axis=0)
    print(velocity_predicted.shape, velocity_target.shape)
    seq_len = velocity_predicted.shape[0]
    for i in range(seq_len):
        vel_f += frechet_distance(velocity_predicted[i, ], velocity_target[i, ])
    return vel_f


def motion_div(predict_joints):

    velocity_predicted = np.diff(predict_joints, axis=0)
    pos_var = np.var(predict_joints)
    vel_var = np.var(velocity_predicted)
    return pos_var, vel_var


def select_aligned(music_beats, motion_beats, tol=6):
    """ Select aligned beats between music and motion.

    For each motion beat, we try to find a one-to-one mapping in the music beats.
    Kwargs:
        music_beats: onehot vector
        motion_beats: onehot vector
        tol: tolerant number of frames [i-tol, i+tol]
    Returns:
        music_beats_aligned: aligned idxs list
        motion_beats_aligned: aligned idxs list
    """
    music_beat_idxs = np.where(music_beats)[0]
    motion_beat_idxs = np.where(motion_beats)[0]

    music_beats_aligned = []
    motion_beats_aligned = []
    accu_inds = []
    for motion_beat_idx in motion_beat_idxs:
        dists = np.abs(music_beat_idxs - motion_beat_idx).astype(np.float32)
        dists[accu_inds] = np.Inf
        ind = np.argmin(dists)

        if dists[ind] > tol:
            continue
        else:
            music_beats_aligned.append(music_beat_idxs[ind])
            motion_beats_aligned.append(motion_beat_idx)
            accu_inds.append(ind)

    music_beats_aligned = np.array(music_beats_aligned)
    motion_beats_aligned = np.array(motion_beats_aligned)
    # print(music_beats_aligned.shape, motion_beats_aligned.shape)
    return music_beats_aligned, motion_beats_aligned

def alignment_score(music_beats, motion_beats, sigma=3):
    """Calculate alignment score between music and motion."""
    if motion_beats.sum() == 0:
        return 0.0

    music_beat_idxs = np.where(music_beats)[0]
    motion_beat_idxs = np.where(motion_beats)[0]

    score_all = []
    for motion_beat_idx in motion_beat_idxs:
        dists = np.abs(music_beat_idxs - motion_beat_idx).astype(np.float32)
        ind = np.argmin(dists)
        score = np.exp(- dists[ind]**2 / 2 / sigma**2)
        score_all.append(score)
    return sum(score_all) / len(score_all) 

def dtw_motion_music(music_beats, motion_beats):
    dist, cost, acc_cost, path = dtw(music_beats.T, motion_beats.T, dist=lambda x, y: norm(x - y, ord=1))
    print('Normalized distance between the two sounds:', dist)
    return cost.T