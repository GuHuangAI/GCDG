from collections import OrderedDict
import pickle
import os
import librosa
import numpy as np
import scipy
import scipy.signal as scisignal
import tqdm
from scipy.spatial.transform import Rotation as R

import warnings

warnings.filterwarnings("ignore")

SR = 60 * 512  # 30720
EPS = 1e-6

FEATURE_DIR = './wav_features_aist'  # total is 13885
os.makedirs(FEATURE_DIR, exist_ok=True)


def music_features_all(path, t_start=0.0, t_dur=None, tempo=120.0, concat=False):
    """The funtion to extract the audio features, save in pkl files.
    Args:
        path
    """
    cache_path = os.path.join(FEATURE_DIR, os.path.basename(path).replace('.wav', '.pkl'))
    data, t_len = music_load(path, t_start, t_dur)
    envelope = music_envelope(data=data)
    mfcc = music_mfcc(data=data)
    chroma = music_chroma(data=data)
    _, peak_onehot = music_peak_onehot(envelope=envelope)
    _, beat_onehot, tempo = music_beat_onehot(envelope=envelope, start_bpm=tempo)
    features = OrderedDict(
        {'envelope': envelope[:, None], 'mfcc': mfcc, 'chroma': chroma, 'peak_onehot': peak_onehot[:, None],
         'beat_onehot': beat_onehot[:, None], })
    # import ipdb; ipdb.set_trace()

    """ mfcc"""
    mean1 = np.mean(features['mfcc'][:, :10], axis=(0))
    std1 = np.std(features['mfcc'][:, :10], axis=(0))
    mean2 = np.mean(features['mfcc'][:, 10:20], axis=(0))
    std2 = np.std(features['mfcc'][:, 10:20], axis=(0))

    features['mfcc'][:, :10] -= mean1
    features['mfcc'][:, :10] /= 2 * std1

    features['mfcc'][:, 10:20] -= mean2
    features['mfcc'][:, 10:20] /= 4 * std2

    """ envelope & chroma"""
    mean3 = np.mean(features['envelope'], axis=(0))
    std3 = np.std(features['envelope'], axis=(0))
    features['envelope'] -= mean3
    features['envelope'] /= std3

    mean4 = np.mean(features['chroma'], axis=(0))
    std4 = np.std(features['chroma'], axis=(0))
    features['chroma'] -= mean4
    features['chroma'] /= std4

    if concat:
        features = np.concatenate([v for k, v in features.items()], axis=1)
        # print('features.shape = {}'.format(features.shape))

    with open(cache_path, 'wb') as f:
        pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)

    features = pickle.load(open(cache_path, 'rb'))
    print('features.shape = {}'.format(features.shape))


def music_load(path, t_start, t_dur):
    """Load raw music data."""
    data, sr = librosa.load(path, sr=SR, offset=t_start, duration=t_dur)
    return data, len(data) / sr


def music_envelope(path=None, data=None):
    """Calculate raw music envelope."""
    assert (path is not None) or (data is not None)
    envelope = librosa.onset.onset_strength(data, sr=SR)  #
    return envelope  # (seq_len,)


def music_mfcc(path=None, data=None, m_mfcc=20):
    """Calculate music feature: mfcc."""
    assert (path is not None) or (data is not None)
    mfcc = librosa.feature.mfcc(data, sr=SR, n_mfcc=m_mfcc, hop_length=512, n_fft=1024)
    return mfcc.T  # (seq_len, 20)


def music_chroma(path=None, data=None, n_chroma=12):
    """Calculate music feature: chroma."""
    assert (path is not None) or (data is not None)
    chroma = librosa.feature.chroma_cens(data, sr=SR, hop_length=512, n_chroma=n_chroma)
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
    peak_idxs = librosa.onset.onset_detect(onset_envelope=envelope.flatten(), sr=SR, hop_length=512)
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
    tempo, beat_idxs = librosa.beat.beat_track(onset_envelope=envelope, sr=SR, hop_length=512,
                                               start_bpm=start_bpm, tightness=tightness)
    beat_onehot = np.zeros_like(envelope, dtype=bool)
    beat_onehot[beat_idxs] = 1
    return envelope, beat_onehot, tempo


def get_music_name(video_name):
    """Get AIST music name for a specific video."""
    splits = video_name.split('_')
    return splits[-2]


def get_tempo(music_name):
    """Get tempo (BPM) for a music by parsing music name."""
    assert len(music_name) == 4
    if music_name[0:3] in ['mBR', 'mPO', 'mLO', 'mMH', 'mLH', 'mWA', 'mKR', 'mJS', 'mJB']:
        return int(music_name[3]) * 10 + 80
    elif music_name[0:3] == 'mHO':
        return int(music_name[3]) * 5 + 110
    else:
        assert False, music_name


if __name__ == '__main__':
    audio_sequence_path = './wav'

    for sequence in tqdm.tqdm(os.listdir(audio_sequence_path)):
        music_name = sequence[:-4]
        # music_name = get_music_name(sequence[:-4])
        music_tempo = get_tempo(sequence[:-4])
        # music_tempo = get_tempo(music_name)
        music_path = os.path.join(audio_sequence_path, f'{sequence}')
        d, dur_ = librosa.load(music_path, sr=SR, offset=0, duration=None)
        music_features = music_features_all(music_path, t_start=0, tempo=music_tempo, concat=True)

