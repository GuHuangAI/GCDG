from collections import OrderedDict
import pickle
import os
from . import _init_paths
import librosa
import numpy as np
import scipy
import scipy.signal as scisignal
import tqdm
from scipy.spatial.transform import Rotation as R
from configs.configs import config

MF_CACHE_DIR = config.MF_CACHE_DIR
FPS = 60
HOP_LENGTH = 512
# SR = 61363
SR = FPS * HOP_LENGTH   # 30720
EPS = 1e-6

os.makedirs(MF_CACHE_DIR, exist_ok=True)

# ===========================================================
# Music Processing Fuctions
# ===========================================================
def music_features_all(path, t_start=0.0, t_dur=None, tempo=120.0, concat=False):
    cache_path = os.path.join(MF_CACHE_DIR, os.path.basename(path).replace('.wav', '.pkl'))
    # if os.path.exists(cache_path):
    #     with open(cache_path, 'rb') as f:
    #         features = pickle.load(f)
    # else:
    # if os.path.exists(cache_path):
    #     features = pickle.load(open(cache_path, 'rb'))
    # else:
    data, t_len = music_load(path, t_start, t_dur)
        # print(data.shape)
    envelope = music_envelope(data=data)
    mfcc = music_mfcc(data=data)
    chroma = music_chroma(data=data)
    _, peak_onehot = music_peak_onehot(envelope=envelope)
    _, beat_onehot, tempo = music_beat_onehot(envelope=envelope, start_bpm=tempo)
    features = OrderedDict({'envelope': envelope[:, None], 'mfcc': mfcc, 'chroma': chroma,
            'peak_onehot': peak_onehot[:, None], 'beat_onehot': beat_onehot[:, None],})
    with open(cache_path, 'wb') as f:
        pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)

    if concat:
        return np.concatenate([v for k, v in features.items()], axis=1)
    else:
        return features


def music_load(path, t_start, t_dur):
    """Load raw music data."""
    data, sr = librosa.load(path, sr=SR, offset=t_start, duration=t_dur)
    return data, len(data)/sr


# envelope特征提取
def music_envelope(path=None, data=None):
    """Calculate raw music envelope."""
    assert (path is not None) or (data is not None)

    # D = librosa.stft(data, hop_length=512, n_fft=512, center=False)
    # melspec = librosa.feature.melspectrogram(S=np.abs(D)**2)
    # envelope = librosa.onset.onset_strength(S=librosa.power_to_db(melspec))
    envelope = librosa.onset.onset_strength(data, sr=SR) # 计算每帧的起始强度
    return envelope # (seq_len,)

# mfcc特征提取
def music_mfcc(path=None, data=None, m_mfcc=20):
    """Calculate music feature: mfcc."""
    assert (path is not None) or (data is not None)

    # D = librosa.stft(data, hop_length=512, n_fft=512, center=False)
    # melspec = librosa.feature.melspectrogram(S=np.abs(D)**2)
    # mfcc = librosa.feature.mfcc(S=librosa.power_to_db(melspec), n_mfcc=m_mfcc)
    mfcc = librosa.feature.mfcc(data, sr=SR, n_mfcc=m_mfcc, hop_length=512, n_fft=1024)
    # print('---mfcc', mfcc.T.shape)

    return mfcc.T  # (seq_len, 20)

# chroma特征提取
def music_chroma(path=None, data=None, n_chroma=12):
    """Calculate music feature: chroma."""
    assert (path is not None) or (data is not None)

    # D = librosa.stft(data, hop_length=512, n_fft=512, center=False)
    # melspec = librosa.feature.melspectrogram(S=np.abs(D)**2)
    # chroma = librosa.feature.chroma_cens(C=librosa.power_to_db(melspec), n_chroma=n_chroma)
    chroma = librosa.feature.chroma_cens(data, sr=SR, hop_length=512, n_chroma=n_chroma)
    # print('---chroma', chroma.T.shape)
    return chroma.T  # (seq_len, 12)

# peak_onehot特征提取
def music_peak_onehot(path=None, envelope=None):
    """Calculate music onset peaks.
    
    Return:
        - envelope: float array with shape of (seq_len,)
        - peak_onehot: one-hot array with shape of (seq_len,)
    """
    assert (path is not None) or (envelope is not None)
    if envelope is None:
        envelope = music_envelope(path=path)
    # peak_idxs = librosa.onset.onset_detect(onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH)
    peak_idxs = librosa.onset.onset_detect(onset_envelope=envelope.flatten(), sr=SR, hop_length=512)
    peak_onehot = np.zeros_like(envelope, dtype=bool)
    peak_onehot[peak_idxs] = 1
    # print('---peak_onehot', peak_onehot.shape)
    return envelope, peak_onehot

# beat_onehot特征提取
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
    # tempo, beat_idxs = librosa.beat.beat_track(onset_envelope=envelope, sr=SR,
    #     start_bpm=start_bpm, tightness=tightness)
    tempo, beat_idxs = librosa.beat.beat_track(onset_envelope=envelope, sr=SR, hop_length=512,
        start_bpm=start_bpm, tightness=tightness)
    beat_onehot = np.zeros_like(envelope, dtype=bool)
    beat_onehot[beat_idxs] = 1
    # print('---beat_onehot', beat_onehot.shape)
    return envelope, beat_onehot, tempo


# ===========================================================
# Motion Processing Fuctions
# ===========================================================
def interp_motion(joints):
    """Interpolate 30FPS motion into 60FPS."""
    # print(joints.shape)
    seq_len = joints.shape[0]
    print(seq_len)
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


# ===========================================================
# Metrics Processing Fuctions
# ===========================================================
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


def calculate_metrics(music_paths, joints_list, bpm_list, tol=12, sigma=3, start_id=0, verbose=False):
    """Calculate metrics for (motion, music) pair"""
    assert len(music_paths) == len(joints_list)
    
    metrics = {
        'beat_coverage': [],
        'beat_hit': [],
        'beat_alignment': [],
        'beat_energy': [],
        'motion_energy': [],
    }

    for music_path, joints, bpm in zip(music_paths, joints_list, bpm_list):
        bpm = 120 if bpm is None else bpm
        print(music_path)
        # extract beats
        # music_envelope, music_beats, tempo = music_beat_onehot(music_path, start_bpm=bpm)
        # music_envelope, music_beats = music_peak_onehot(music_path)

        music_features = music_features_all(music_path, tempo=bpm)
        music_envelope, music_beats = music_features['envelope'], music_features['beat_onehot']
        motion_envelope, motion_beats, motion_beats_energy = motion_peak_onehot(joints)
        
        end_id = min(motion_envelope.shape[0], music_envelope.shape[0])

        music_envelope = music_envelope[start_id:end_id]
        music_beats = music_beats[start_id:end_id]
        motion_envelope = motion_envelope[start_id:end_id]
        motion_beats = motion_beats[start_id:end_id]
        motion_beats_energy = motion_beats_energy[start_id:end_id]
        
        # alignment
        print('***before***', music_beats.shape, motion_beats.shape)

        music_beats_aligned, motion_beats_aligned = select_aligned(music_beats, motion_beats, tol=tol)
        
        # print('***after***', music_beats_aligned.shape, motion_beats_aligned.shape)
        print('***after***')
        print(music_beats_aligned)
        print(motion_beats_aligned)

        # metrics
        metrics['beat_coverage'].append(
            motion_beats.sum() / (music_beats.sum() + EPS))        
        metrics['beat_hit'].append(
            len(motion_beats_aligned) / (motion_beats.sum() + EPS))
        metrics['beat_alignment'].append(
            alignment_score(music_beats, motion_beats, sigma=sigma))
        metrics['beat_energy'].append(
            motion_beats_energy[motion_beats].mean() if motion_beats.sum() > 0 else 0.0)
        metrics['motion_energy'].append(
            motion_envelope.mean())
        
    for k, v in metrics.items():
        metrics[k] = sum(v) / len(v)
        if verbose:
            print (f'{k}: {metrics[k]:.3f}')
    return metrics


# ===========================================================
# 旋转角和旋转矩阵
# ===========================================================
def transform_rot_representation(rot, input_type='vec',out_type='mat'):
    '''
    make transformation between different representation of 3D rotation
    input_type / out_type (np.array):
        'mat': rotation matrix (3*3)
        'quat': quaternion (4)
        'vec': rotation vector (3)
        'euler': Euler degrees in x,y,z (3)
    '''
    if input_type=='mat':
        r = R.from_matrix(rot)
    elif input_type=='quat':
        r = R.from_quat(rot)
    elif input_type =='vec':
        r = R.from_rotvec(rot)
    elif input_type =='euler':
        if rot.max()<4:
            rot = rot*180/np.pi
        r = R.from_euler('xyz',rot, degrees=True)
    
    if out_type=='mat':
        out = r.as_matrix()
    elif out_type=='quat':
        out = r.as_quat()
    elif out_type =='vec':
        out = r.as_rotvec()
    elif out_type =='euler':
        out = r.as_euler('xyz', degrees=False)
    return out








