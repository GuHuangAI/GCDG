# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.

import os
import sys
import json
import random
import argparse
import essentia
import tqdm
import pickle
import essentia.streaming
from essentia.standard import *
import librosa
import numpy as np
from extractor import FeatureExtractor
import warnings
warnings.filterwarnings("ignore")

FEATURE_DIR = './audio_sequence_features' # total is 13885
os.makedirs(FEATURE_DIR, exist_ok=True)

extractor = FeatureExtractor()

def extract_acoustic_feature(path, t_start=0.0, t_dur=None, tempo=120.0):
    print('---------- Extract features from raw audio ----------')

    cache_path = os.path.join(FEATURE_DIR, os.path.basename(path).replace('.wav', '.pkl'))
    sr = 60 * 512 
    loader = essentia.standard.MonoLoader(filename=path, sampleRate=sr)
    audio = loader()
    audio = np.array(audio).T
    melspe_db = extractor.get_melspectrogram(audio, sr)
    mfcc = extractor.get_mfcc(melspe_db)
    mfcc_delta = extractor.get_mfcc_delta(mfcc)

    audio_harmonic, audio_percussive = extractor.get_hpss(audio)
    chroma_cqt = extractor.get_chroma_cqt(audio_harmonic, sr)

    onset_env = extractor.get_onset_strength(audio_percussive, sr)
    tempogram = extractor.get_tempogram(onset_env, sr)
    onset_beat = extractor.get_onset_beat(onset_env, sr)

    onset_env = onset_env.reshape(1, -1)

    feature = np.concatenate([
        mfcc,
        mfcc_delta,
        chroma_cqt,
        onset_env,
        onset_beat,
        tempogram
    ], axis=0)

    feature = feature.transpose(1, 0)
    print(f'acoustic feature -> {feature.shape}')

    with open(cache_path, 'wb') as f:
        pickle.dump(feature, f, protocol=pickle.HIGHEST_PROTOCOL)

def align(musics, dances):
    print('---------- Align the frames of music and dance ----------')
    assert len(musics) == len(dances), \
        'the number of audios should be equal to that of videos'
    new_musics=[]
    new_dances=[]
    for i in range(len(musics)):
        min_seq_len = min(len(musics[i]), len(dances[i]))
        print(f'music -> {np.array(musics[i]).shape}, ' +
              f'dance -> {np.array(dances[i]).shape}, ' +
              f'min_seq_len -> {min_seq_len}')
        del musics[i][min_seq_len:]
        del dances[i][min_seq_len:]
        new_musics.append([musics[i][j] for j in range(min_seq_len) if j%2==0])
        new_musics.append([musics[i][j] for j in range(min_seq_len) if j%2==1])
        
        new_dances.append([dances[i][j] for j in range(min_seq_len) if j%2==0])
        new_dances.append([dances[i][j] for j in range(min_seq_len) if j%2==1])
    return new_musics, new_dances



def save(args, musics, dances, inner_dir):
    print('---------- Save to text file ----------')
    fnames = sorted(os.listdir(os.path.join(args.input_dance_dir,inner_dir)))
    # fnames = fnames[:20]  # for debug
    assert len(fnames)*2 == len(musics) == len(dances), 'alignment'

    train_idx, test_idx = split_data(fnames)
    train_idx = sorted(train_idx)
    print(f'train ids: {[fnames[idx] for idx in train_idx]}')
    test_idx = sorted(test_idx)
    print(f'test ids: {[fnames[idx] for idx in test_idx]}')

    print('---------- train data ----------')

    for idx in train_idx:
        for i in range(2):
            with open(os.path.join(args.train_dir, f'{inner_dir+"_"+fnames[idx]+"_"+str(i)}.json'), 'w') as f:
                sample_dict = {
                    'id': fnames[idx]+"_"+str(i),
                    'music_array': musics[idx*2+i],
                    'dance_array': dances[idx*2+i]
                }
                json.dump(sample_dict, f)

    print('---------- test data ----------')
    for idx in test_idx:
        for i in range(2):
            with open(os.path.join(args.test_dir, f'{inner_dir+"_"+fnames[idx]+"_"+str(i)}.json'), 'w') as f:
                sample_dict = {
                    'id': fnames[idx]+"_"+str(i),
                    'music_array': musics[idx*2+i],
                    'dance_array': dances[idx*2+i]
                }
                json.dump(sample_dict, f)


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
    # audio_sequence_path = '/export2/home/lsy/TTA_data/audio_sequence'
    audio_path = r'G:\database\TTA_data\ext_audio'
    for sequence in tqdm.tqdm(os.listdir(audio_path)):
        music_name = get_music_name(sequence[:25])
        music_tempo = get_tempo(music_name)
        music_path = os.path.join(audio_path, f'{sequence}')
        music_features = extract_acoustic_feature(music_path, t_start=0, tempo=music_tempo)
    pass