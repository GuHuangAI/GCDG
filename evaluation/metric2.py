from collections import OrderedDict
import pickle
import os
import librosa
import numpy as np
import scipy
import scipy.signal as scisignal
import tqdm
import math
from dtw.dtw import dtw
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R
from scipy import linalg
import random
import torch
from evaluation.kinetic import extract_kinetic_features
from evaluation.manual import extract_manual_features
import itertools

FPS = 60
HOP_LENGTH = 512
SR = FPS * HOP_LENGTH
EPS = 1e-6
CACHE_DIR = '/export2/home/lsy/TTA_data/audio_sequence_features_new/err/'

# ===========================================================
# FID and Diversity
# ===========================================================
def FID(predict_joints, target_joints, classifier, genres_label, num_sample=1000, seq_len=100):
    '''
        predict_joints/target_joints [List] len = num; each element is a np.ndarray (seq_len, num_joints, 3)
        num is the number of sequence
        seq_len: frames
        note the seq_len of targets are different
    '''
    num = len(predict_joints)
    # seq_len, num_joints, _ = predict_joints[0].shape
    # sample 1000 1-s seqs #
    sample_pred_joints = np.zeros((num_sample, seq_len, 24*3)) # num_joints=24
    sample_targ_joints = np.zeros((num_sample, seq_len, 24*3))
    label = np.zeros((num_sample, ))
    for i in range(num_sample):
        seq_ind = random.randint(0, num-1)
        predict_joint = predict_joints[seq_ind]
        target_joint = target_joints[seq_ind]
        seq_len_pred = predict_joint.shape[0]
        seq_len_targ = target_joint.shape[0]
        predict_joint = predict_joint.reshape((seq_len_pred, -1))
        target_joint = target_joint.reshape((seq_len_targ, -1))
        frame_ind = random.randint(0, seq_len_targ-seq_len-1)
        # frame_ind2 = random.randint(0, seq_len_pred - seq_len - 1)
        sample_pred_joints[i, :, :] = predict_joint[frame_ind: frame_ind + seq_len, :]
        sample_targ_joints[i, :, :] = target_joint[frame_ind: frame_ind + seq_len, :]
        # sample_pred_joints2[i, :, :] = predict_joint[frame_ind2: frame_ind2 + seq_len, :]
        label[i] = genres_label[seq_ind]

    sample_pred_joints2 = np.zeros((num, seq_len, 24 * 3))  # for diversity
    sample_targ_joints2 = np.zeros((num, seq_len, 24 * 3))  # for diversity
    for i in range(num):
        predict_joint = predict_joints[i]
        target_joint = target_joints[i]
        seq_len_pred = predict_joint.shape[0]
        seq_len_targ = target_joint.shape[0]
        predict_joint = predict_joint.reshape((seq_len_pred, -1))
        target_joint = target_joint.reshape((seq_len_targ, -1))
        frame_ind1 = random.randint(0, 1200 - seq_len - 1)
        frame_ind2 = random.randint(0, seq_len_targ - seq_len - 1)
        sample_pred_joints2[i, :, :] = predict_joint[frame_ind1: frame_ind1 + seq_len, :]
        sample_targ_joints2[i, :, :] = target_joint[frame_ind2: frame_ind2 + seq_len, :]

    # calculate fid
    device = torch.device("cuda")
    bs = 10
    data_pred = torch.from_numpy(sample_pred_joints)
    data_tgt = torch.from_numpy(sample_targ_joints)
    # print('data_tgt', data_tgt.shape)
    features1, features2, pred_acc, real_acc = get_features(data_pred, data_tgt, classifier=classifier, label=label, bs=bs, device=device)
    mu_act1, sigma_act1 = compute_act_mean_std(features1)
    mu_act2, sigma_act2 = compute_act_mean_std(features2)
    fid = _compute_FID(mu_act1, mu_act2, sigma_act1, sigma_act2)

    sample_pred_joints2 = torch.from_numpy(sample_pred_joints2)
    sample_targ_joints2 = torch.from_numpy(sample_targ_joints2)
    div = get_diversity(sample_pred_joints2, classifier=classifier, bs=bs, device=device)
    real_div = get_diversity(sample_targ_joints2, classifier=classifier, bs=bs, device=device)
    print('real_div:', real_div)

    return fid, div, pred_acc, real_acc

def get_features(data_pred, data_tgt, classifier, label, bs, device):
    num_batches = data_pred.shape[0]//bs
    classifier = classifier.to(device)
    classifier.eval()
    features1 = np.zeros((data_pred.shape[0], classifier.hidden_size))
    features2 = np.zeros((data_tgt.shape[0], classifier.hidden_size))
    pred_acc = 0.
    real_acc = 0.
    for batch_ind in range(num_batches):
        data_act1 = data_pred[batch_ind*bs:(batch_ind+1)*bs, :, :].float().to(device)
        data_act2 = data_tgt[batch_ind*bs:(batch_ind+1)*bs, :, :].float().to(device)
        batch_label = label[batch_ind*bs:(batch_ind+1)*bs]
        logits1, fea1 = classifier(data_act1)
        logits2, fea2 = classifier(data_act2)
        features1[batch_ind * bs:(batch_ind+1)*bs, :] = fea1.cpu().data.numpy()
        features2[batch_ind * bs:(batch_ind+1)*bs, :] = fea2.cpu().data.numpy()
        pred_class = torch.argmax(logits1, dim=1).cpu().data.numpy()
        real_class = torch.argmax(logits2, dim=1).cpu().data.numpy()
        pred_acc += (pred_class == batch_label).sum()
        real_acc += (real_class == batch_label).sum()
    pred_acc /= data_pred.shape[0]
    real_acc /= data_pred.shape[0]
    return features1, features2, pred_acc, real_acc

def get_diversity(joints, classifier, bs, device, num_combination=780):
    num_batches = joints.shape[0] // bs
    classifier = classifier.to(device)
    classifier.eval()
    features1 = np.zeros((joints.shape[0], classifier.hidden_size))
    # features2 = np.zeros((data_tgt.shape[0], classifier.hidden_size))
    # pred_acc = 0.
    # real_acc = 0.
    for batch_ind in range(num_batches):
        data_act1 = joints[batch_ind * bs:(batch_ind + 1) * bs, :, :].float().to(device)
        # data_act2 = data_tgt[batch_ind * bs:(batch_ind + 1) * bs, :, :].float().to(device)
        # batch_label = label[batch_ind * bs:(batch_ind + 1) * bs]
        logits1, fea1 = classifier(data_act1)
        # logits2, fea2 = classifier(data_act2)
        features1[batch_ind * bs:(batch_ind + 1) * bs, :] = fea1.cpu().data.numpy()
        # features2[batch_ind * bs:(batch_ind + 1) * bs, :] = fea2.cpu().data.numpy()
        # pred_class = torch.argmax(logits1, dim=1).cpu().data.numpy()
        # real_class = torch.argmax(logits2, dim=1).cpu().data.numpy()
        # pred_acc += (pred_class == batch_label).sum()
        # real_acc += (real_class == batch_label).sum()

    # betnums = range(features1.shape[0])
    div = 0.
    com = list(itertools.combinations(range(joints.shape[0]), 2))
    for j in range(len(com)):
        ind1, ind2 = com[j]
        dist = np.linalg.norm(features1[ind1] - features1[ind2])
        # betnum = random.sample(betnums, 2)
        # dist = np.linalg.norm(features1[betnum[0]] - features1[betnum[1]])
        div += dist
    div = div/len(com)
    return div

def norm_fea(features):
    # normalize the scale
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0) + 1e-10
    features = (features - mean) / std
    # feature_list2 = (feature_list2 - mean) / std
    return features

def norm_fea2(features_real, features_pred):
    # normalize the scale
    mean = np.mean(features_real, axis=0)
    std = np.std(features_real, axis=0) + 1e-10
    features_real = (features_real - mean) / std
    features_pred = (features_pred - mean) / std
    return features_real, features_pred

def FID_Dist_kg(predict_joints, target_joints):
    '''
    predict_joints [List]: len = 40; each element is a np.ndarray (seq_len_pred, num_joints, 3)
    target_joints [List]: len = 40; each element is a np.ndarray (seq_len_tgt, num_joints, 3)
    seq_len_pred = 1200 (20 seconds)
    seq_len_tgt is not equal to seq_len_pred
    '''
    num = len(predict_joints)
    kinetic_features_pred = np.zeros((num, 72))
    kinetic_features_targ = np.zeros((num, 72))
    geometric_features_pred = np.zeros((num, 32))
    geometric_features_targ = np.zeros((num, 32))
    for i in range(num):
        predict_joint = predict_joints[i]
        target_joint = target_joints[i]
        kinetic_features_pred[i, :] = extract_kinetic_features(predict_joint)
        kinetic_features_targ[i, :] = extract_kinetic_features(target_joint)
        geometric_features_pred[i, :] = extract_manual_features(predict_joint)
        geometric_features_targ[i, :] = extract_manual_features(target_joint)

    # kinetic_features_pred = norm_fea(kinetic_features_pred)
    # kinetic_features_targ = norm_fea(kinetic_features_targ)
    kinetic_features_targ, kinetic_features_pred = norm_fea2(kinetic_features_targ, kinetic_features_pred)
    # geometric_features_pred = norm_fea(geometric_features_pred)
    # geometric_features_targ = norm_fea(geometric_features_targ)
    geometric_features_targ, geometric_features_pred = norm_fea2(geometric_features_targ, geometric_features_pred)

    mu_act1, sigma_act1 = compute_act_mean_std(kinetic_features_pred)
    mu_act2, sigma_act2 = compute_act_mean_std(kinetic_features_targ)
    fid_k = _compute_FID(mu_act1, mu_act2, sigma_act1, sigma_act2)

    mu_act1, sigma_act1 = compute_act_mean_std(geometric_features_pred)
    mu_act2, sigma_act2 = compute_act_mean_std(geometric_features_targ)
    fid_g = _compute_FID(mu_act1, mu_act2, sigma_act1, sigma_act2)

    # calculate dist_kg
    dist_k, dist_g = 0., 0.
    real_dist_k, real_dist_g = 0., 0.
    com = list(itertools.combinations(range(num), 2))
    for j in range(len(com)):
        ind1, ind2 = com[j]
        dist1 = np.linalg.norm(kinetic_features_pred[ind1] - kinetic_features_pred[ind2])
        dist2 = np.linalg.norm(geometric_features_pred[ind1] - geometric_features_pred[ind2])
        dist_k += dist1
        dist_g += dist2
        dist1 = np.linalg.norm(kinetic_features_targ[ind1] - kinetic_features_targ[ind2])
        dist2 = np.linalg.norm(geometric_features_targ[ind1] - geometric_features_targ[ind2])
        real_dist_k += dist1
        real_dist_g += dist2
    dist_k /= len(com)
    dist_g /= len(com)
    real_dist_k /= len(com)
    real_dist_g /= len(com)

    return fid_k, fid_g, dist_k, dist_g, real_dist_k, real_dist_g

# ===========================================================
# Frechet Distance
# ===========================================================
def euc_dist(pt1, pt2):
    # return math.sqrt((pt2[0] - pt1[0]) * (pt2[0] - pt1[0]) + (pt2[1] - pt1[1]) * (pt2[1] - pt1[1]))
    return np.sqrt(sum(np.power((pt1 - pt2), 2)))


def _c(ca, i, j, P, Q):
    if ca[i, j] > -1:
        return ca[i, j]
    elif i == 0 and j == 0:
        ca[i, j] = euc_dist(P[0], Q[0])
    elif i > 0 and j == 0:
        ca[i, j] = max(_c(ca, i - 1, 0, P, Q), euc_dist(P[i], Q[0]))
    elif i == 0 and j > 0:
        ca[i, j] = max(_c(ca, 0, j - 1, P, Q), euc_dist(P[0], Q[j]))
    elif i > 0 and j > 0:
        ca[i, j] = max(min(_c(ca, i - 1, j, P, Q), _c(ca, i - 1, j - 1, P, Q), _c(ca, i, j - 1, P, Q)),
                       euc_dist(P[i], Q[j]))
    else:
        ca[i, j] = float("inf")
    return ca[i, j]


def frechet_distance(P, Q):
    ca = np.ones((len(P), len(Q)))
    ca = np.multiply(ca, -1)
    return _c(ca, len(P) - 1, len(Q) - 1, P, Q)  # ca鏄痑*b鐨勭煩闃?3*4),2,3

def FD(predict_joints, target_joints):
    '''
    predict_joints/target_joints: [num, seq_len, N*3]
    num is the number of sequence
    seq_len: 240 frames
    N: the number of joints
    '''
    ### sample 1000 1-s seqs
    sample_pred_joints = np.zeros((predict_joints.shape[0], 60, predict_joints.shape[2]))
    sample_targ_joints = np.zeros((target_joints.shape[0], 60, target_joints.shape[2]))
    for i in range(predict_joints.shape[0]):
        index = random.randint(0,179)
        sample_pred_joints[i, :, :] = predict_joints[i, index: index + 60, :]
        sample_targ_joints[i, :, :] = target_joints[i, index: index + 60, :]

    ### calculate fd
    num = sample_pred_joints.shape[0]
    act1 = sample_pred_joints.reshape((num,-1))
    act2 = sample_targ_joints.reshape((num, -1))
    mu_act1, sigma_act1 = compute_act_mean_std(act1)
    mu_act2, sigma_act2 = compute_act_mean_std(act2)
    pos_fd = _compute_FID(mu_act1, mu_act2, sigma_act1, sigma_act2)

    velocity_predicted = np.zeros_like(sample_pred_joints, dtype=np.float32)
    velocity_predicted[:, 1:, :] = sample_pred_joints[:, 1:, :] - sample_pred_joints[:, :-1, :]
    velocity_target = np.zeros_like(sample_targ_joints, dtype=np.float32)
    velocity_target[:, 1:, :] = sample_targ_joints[:, 1:, :] - sample_targ_joints[:, :-1, :]
    # num = sample_pred_joints.shape[0]
    act1 = velocity_predicted.reshape((num, -1))
    act2 = velocity_target.reshape((num, -1))
    mu_act1, sigma_act1 = compute_act_mean_std(act1)
    mu_act2, sigma_act2 = compute_act_mean_std(act2)
    vel_fd = _compute_FID(mu_act1, mu_act2, sigma_act1, sigma_act2)
    return pos_fd, vel_fd


def vel_frechet(predict_joints, target_joints):
    '''
        predict_joints/target_joints: [num, seq_len, N*3]
        num is the number of sequence
        seq_len: 60 frames
        N: the number of joints
    '''
    velocity_predicted = np.zeros_like(predict_joints, dtype=np.float32)
    velocity_predicted[:, 1:, :] = predict_joints[:, 1:, :] - predict_joints[:, :-1, :]
    velocity_target = np.zeros_like(target_joints, dtype=np.float32)
    velocity_target[:, 1:, :] = target_joints[:, 1:, :] - target_joints[:, :-1, :]
    num = predict_joints.shape[0]
    act1 = velocity_predicted.reshape((num, -1))
    act2 = velocity_target.reshape((num, -1))
    mu_act1, sigma_act1 = compute_act_mean_std(act1)
    mu_act2, sigma_act2 = compute_act_mean_std(act2)
    vel_fd = _compute_FID(mu_act1, mu_act2, sigma_act1, sigma_act2)
    return vel_fd


def motion_div(predict_joints):
    # velocity_predicted = np.diff(predict_joints, axis=0)
    num = predict_joints.shape[0]
    velocity_predicted = np.zeros_like(predict_joints, dtype=np.float32)
    velocity_predicted[:, 1:, :] = (predict_joints[:, 1:, :] - predict_joints[:, :-1, :]) * 60
    predict_joints = predict_joints.reshape((num, -1))
    velocity_predicted = velocity_predicted.reshape((num, -1))
    pos_var = np.var(predict_joints, axis=0)
    vel_var = np.var(velocity_predicted, axis=0)
    return pos_var.mean(), vel_var.mean()

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
        score = np.exp(- dists[ind] ** 2 / 2 / sigma ** 2)
        score_all.append(score)
    return sum(score_all) / len(score_all)

def dtw_motion_music(music_beats, motion_beats):
    '''
    music_beats: onehot vector
    motion_beats: onehot vector
    '''
    dist, cost, acc_cost, path = dtw(music_beats.reshape(-1, 1), motion_beats.reshape(-1, 1), dist=lambda x, y: abs(x - y))
    #print('Normalized distance between the two sounds:', dist)
    return dist/len(path[0])

# compute the activation's statistics: mean and std
def compute_act_mean_std(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

# compute FID
def _compute_FID(mu1, mu2, sigma1, sigma2,eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    FID = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    return FID


def calculate_metrics(
        music_paths, joints_list, joints_list2, bpm_list, classifier, genres_label,
        tol=12, sigma=3, start_id=0, verbose=False):
    """Calculate metrics for (motion, music) pair
    joints_list [List]: for predict; each element is a np.ndarray (seq_len, num_joints, 3)
    joints_list2 [List]: for target; each element is a np.ndarray (seq_len, num_joints, 3)
    note the seq_len for target are different
    """
    assert len(music_paths) == len(joints_list)

    metrics = {
        'beat_hit': [],
        'beat_alignment': [],
        # 'beat_energy': [],
        # 'motion_energy': [],
        # 'dtw_score': [],
    }
    for music_path, joints, joints2, bpm in tqdm.tqdm(zip(music_paths, joints_list, joints_list2, bpm_list), desc='Calculating metrics:'):
        bpm = 120 if bpm is None else bpm
        # print(bpm)
        # extract beats
        # music_envelope, music_beats, tempo = music_beat_onehot(music_path, start_bpm=bpm)
        # music_envelope, music_beats = music_peak_onehot(music_path)
        music_features = music_features_all(music_path, tempo=bpm)
        music_envelope, music_beats = music_features['envelope'][120:], music_features['beat_onehot'][120:]
        motion_envelope, motion_beats, motion_beats_energy = motion_peak_onehot(joints)
        # motion_envelope2, motion_beats2, motion_beats_energy2 = motion_peak_onehot(joints2)

        end_id = min(motion_envelope.shape[0], music_envelope.shape[0])

        # music_envelope = music_envelope[start_id:end_id]
        music_beats = music_beats[start_id:end_id]
        # motion_envelope = motion_envelope[start_id:end_id]
        motion_beats = motion_beats[start_id:end_id]
        # motion_beats_energy = motion_beats_energy[start_id:end_id]

        # alignment
        music_beats_aligned, motion_beats_aligned = select_aligned(
            music_beats, motion_beats, tol=tol)

        # metrics
        metrics['beat_hit'].append(
            len(motion_beats_aligned) / (motion_beats.sum() + EPS))
        metrics['beat_alignment'].append(
            alignment_score(music_beats, motion_beats, sigma=sigma))

    for k, v in metrics.items():
        metrics[k] = sum(v) / len(v)
        # if verbose:
        #     print(f'{k}: {metrics[k]:.3f}')
    # joints_numpy = np.stack(joints_list, axis=0)
    # joints_numpy2 = np.stack(joints_list2, axis=0)
    # print(joints_numpy.shape, joints_numpy2.shape)
    # num_sample, seq_len, num_joint, dim_joint = joints_numpy.shape
    # joints_numpy = joints_numpy.reshape((num_sample, seq_len, -1))
    # joints_numpy2 = joints_numpy2.reshape((num_sample, seq_len, -1))
    # print('Calculating var......')
    # print(metrics)
    print('Calculating fid, div, acc (mean of 1 times)......')
    fid, div, pred_acc, real_acc = 0., 0., 0., 0.
    mean_times = 1
    for i in range(mean_times):
        results = FID(joints_list, joints_list2, classifier, genres_label, num_sample=1000)
        fid += results[0]
        div += results[1]
        pred_acc += results[2]
        real_acc += results[3]
    fid /= mean_times
    div /= mean_times
    pred_acc /= mean_times
    real_acc /= mean_times
    # vel_fd = vel_frechet(joints_numpy, joints_numpy2)
    metrics['fid'] = fid
    metrics['div'] = div
    metrics['pred_acc'] = pred_acc
    metrics['real_acc'] = real_acc

    print('Calculating fid_k, fid_g, dist_k, dist_g......')
    fid_k, fid_g, dist_k, dist_g, real_dist_k, real_dist_g = FID_Dist_kg(joints_list, joints_list2)
    print('real_dist_k, real_dist_g: {}, {}'.format(real_dist_k, real_dist_g))
    metrics['fid_k'] = fid_k
    metrics['fid_g'] = fid_g
    metrics['dist_k'] = dist_k
    metrics['dist_g'] = dist_g
    for k, v in metrics.items():
        if verbose:
            print(f'{k}: {metrics[k]:.3f}')
    return metrics

# ===========================================================
# Music Processing Fuctions
# ===========================================================
def music_features_all(path, tempo=120.0, concat=False):
    cache_path = os.path.join(
        CACHE_DIR, os.path.basename(path).replace('.wav', '.pkl'))
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            features = pickle.load(f)
    else:
        data = music_load(path)
        envelope = music_envelope(data=data)

        # tempogram = music_tempogram(envelope=envelope)
        mfcc = music_mfcc(data=data)
        chroma = music_chroma(data=data)
        _, peak_onehot = music_peak_onehot(envelope=envelope)
        _, beat_onehot, _ = music_beat_onehot(envelope=envelope, start_bpm=tempo)

        features = OrderedDict({
            'envelope': envelope[:, None],
            # 'tempogram': tempogram,
            'mfcc': mfcc,
            'chroma': chroma,
            'peak_onehot': peak_onehot,
            'beat_onehot': beat_onehot,
        })
        # with open(cache_path, 'wb') as f:
        #     pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)

    if concat:
        return np.concatenate([v for k, v in features.items()], axis=1)
    else:
        return features


def music_load(path):
    """Load raw music data."""
    data, _ = librosa.load(path, sr=SR)
    return data


def music_envelope(path=None, data=None):
    """Calculate raw music envelope."""
    assert (path is not None) or (data is not None)
    if data is None:
        data = music_load(path)
    envelope = librosa.onset.onset_strength(data, sr=SR)
    return envelope  # (seq_len,)


def music_tempogram(path=None, envelope=None, win_length=384):
    """Calculate music feature: tempogram."""
    assert (path is not None) or (envelope is not None)
    if envelope is None:
        envelope = music_envelope(path)
    tempogram = librosa.feature.tempogram(
        onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH,
        win_length=win_length)
    return tempogram.T  # (seq_len, 384)


def music_mfcc(path=None, data=None, m_mfcc=20):
    """Calculate music feature: mfcc."""
    assert (path is not None) or (data is not None)
    if data is None:
        data = music_load(path)
    mfcc = librosa.feature.mfcc(data, sr=SR, n_mfcc=m_mfcc)
    return mfcc.T  # (seq_len, 20)


def music_chroma(path=None, data=None, n_chroma=12):
    """Calculate music feature: chroma."""
    assert (path is not None) or (data is not None)
    if data is None:
        data = music_load(path)
    chroma = librosa.feature.chroma_cens(
        data, sr=SR, hop_length=HOP_LENGTH, n_chroma=n_chroma)
    return chroma.T  # (seq_len, 12)


def music_peak_onehot(path=None, envelope=None):
    """Calculate music onset peaks.

    Return:
        - envelope: float array with shape of (seq_len,)
        - peak_onehot: one-hot array with shape of (seq_len,)
    """
    assert (path is not None) or (envelope is not None)
    if envelope is None:
        envelope = music_envelope(path=path)
    peak_idxs = librosa.onset.onset_detect(
        onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH)
    peak_onehot = np.zeros_like(envelope, dtype=bool)
    peak_onehot[peak_idxs] = 1
    return envelope, peak_onehot


def music_beat_onehot(
        path=None, envelope=None, start_bpm=120.0, tightness=100):
    """Calculate music beats.

    Return:
        - envelope: float array with shape of (seq_len,)
        - beat_onehot: one-hot array with shape of (seq_len,)
    """
    assert (path is not None) or (envelope is not None)
    if envelope is None:
        envelope = music_envelope(path=path)
    tempo, beat_idxs = librosa.beat.beat_track(
        onset_envelope=envelope, sr=SR, hop_length=HOP_LENGTH,
        start_bpm=start_bpm, tightness=tightness)
    beat_onehot = np.zeros_like(envelope, dtype=bool)
    beat_onehot[beat_idxs] = 1
    return envelope, beat_onehot, tempo

# ===========================================================
# Motion Processing Fuctions
# ===========================================================
def interp_motion(joints):
    """Interpolate 30FPS motion into 60FPS."""
    seq_len = joints.shape[0]
    x = np.arange(0, seq_len)
    fit = scipy.interpolate.interp1d(x, joints, axis=0, kind='cubic')
    joints = fit(np.linspace(0, seq_len-1, 2*seq_len))
    return joints


def motion_peak_onehot(joints):
    """Calculate motion beats.
    Kwargs:
        joints: [nframes, njoints, 3]
    Returns:
        - envelope: motion energy.
        - peak_onhot: motion beats.
        - peak_energy: motion beats energy.
    """
    # Calculate velocity.
    velocity = np.zeros_like(joints, dtype=np.float32)
    velocity[1:] = joints[1:] - joints[:-1]
    velocity_norms = np.linalg.norm(velocity, axis=2)
    envelope = np.sum(velocity_norms, axis=1)  # (seq_len,)

    # Find local minima in velocity -- beats
    peak_idxs = scisignal.argrelextrema(envelope, np.less, axis=0, order=10) # 10 for 60FPS
    peak_onehot = np.zeros_like(envelope, dtype=bool)
    peak_onehot[peak_idxs] = 1

    # Second-derivative of the velocity shows the energy of the beats
    derivative = np.zeros_like(envelope, dtype=np.float32)
    derivative[2:] = envelope[2:] - envelope[1:-1]
    derivative2 = np.zeros_like(envelope, dtype=np.float32)
    derivative2[3:] = derivative[3:] - derivative[2:-1]
    peak_energy = np.gradient(np.gradient(envelope)) # (seq_len,)

    # optimize peaks
    # peak_onehot[peak_energy<0.5] = 0
    return envelope, peak_onehot, peak_energy

if __name__ == '__main__':
    x1 = np.random.random((1200, 24, 3))
    fea_1 = extract_kinetic_features(x1)
    x2 = np.random.random((10, 240, 72))
    _,_ = FD(x1, x2)
    # pos_fd = pos_frechet(x1, x2)
    # vel_fd = vel_frechet(x1, x2)
    music_path = r'G:\database\AIST\audio_sequence\gWA_sFM_c01_d27_mWA5_ch20_25.wav'
    music_features = music_features_all(music_path, tempo=120)
    import timm
    pass