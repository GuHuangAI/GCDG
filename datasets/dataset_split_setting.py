"""AIST++ train/val/test set split setting."""
import os
import random
import re

MUSIC_ID_TESTVAL = '(mBR0|mPO1|mLO2|mMH3|mLH4|mHO5|mWA0|mKR2|mJS3|mJB5)'
MOTION_GEN_ID_TESTVAL = '.*_sBM_.*_(mBR|mPO|mLO|mMH|mLH|mHO|mWA|mKR|mJS|mJB).*_(ch01|ch02)'
MOTION_GEN_ID_TESTVAL_PAIRED = '.*_sBM_.*_(mBR0|mPO1|mLO2|mMH3|mLH4|mHO5|mWA0|mKR2|mJS3|mJB5)_(ch01|ch02)'
MOTION_GEN_ID_TESTVAL_UNPAIRED = '.*_sBM_.*_(mBR5|mPO5|mLO5|mMH0|mLH0|mHO0|mWA5|mKR5|mJS0|mJB0)_(ch01|ch02)'

MOTION_PRED_ID_VAL = '.*_ch01'
MOTION_PRED_ID_TEST = '.*_ch02'


def get_testval_music_id(video_name):
  music_id = video_name.split('_')[-2]
  if 'mBR' in music_id:
    return 'mBR0'
  elif 'mPO' in music_id:
    return 'mPO1'
  elif 'mLO' in music_id:
    return 'mLO2'
  elif 'mMH' in music_id:
    return 'mMH3'
  elif 'mLH' in music_id:
    return 'mLH4'
  elif 'mHO' in music_id:
    return 'mHO5'
  elif 'mWA' in music_id:
    return 'mWA0'
  elif 'mKR' in music_id:
    return 'mKR2'
  elif 'mJS' in music_id:
    return 'mJS3'
  elif 'mJB' in music_id:
    return 'mJB5'
  else:
    assert False, video_name


def get_split(video_names, task, subset, **kwargs):
  assert task in ['generation', 'prediction']
  assert subset in ['train', 'val', 'test', 'all']

  split = {
      'video_names': [],
      'music_names': [],
      'rng_seed': 58 if subset == 'test' else 42,
      'seq_seed_len': kwargs['seq_seed_len'] if 'seq_seed_len' in kwargs else None,
      'is_paired': kwargs['is_paired'] if 'is_paired' in kwargs else None,
  }

  if task == 'prediction' and subset == 'val':
    split['video_names'] = [
        video_name for video_name in video_names
        if re.match(MOTION_PRED_ID_VAL, video_name)]

  elif task == 'prediction' and subset == 'test':
    split['video_names'] = [
        video_name for video_name in video_names
        if re.match(MOTION_PRED_ID_TEST, video_name)]

  elif task == 'prediction' and subset == 'train':
    split['video_names'] = [
        video_name for video_name in video_names
        if not re.match(MOTION_PRED_ID_VAL, video_name) and \
           not re.match(MOTION_PRED_ID_TEST, video_name)]

  elif task == 'generation' and (subset == 'val' or subset == 'test'):
    assert split['is_paired'] in [True, False]
    if split['is_paired']:
      split['video_names'] = [
          video_name for video_name in video_names
          if re.match(MOTION_GEN_ID_TESTVAL_PAIRED, video_name)]
      split['music_names'] = [
          video_name for video_name in split['video_names']]

      # split['music_names'] = [
      #     video_name.split('_')[-2] for video_name in split['video_names']]
    else:
      split['video_names'] = [
          video_name for video_name in video_names
          if re.match(MOTION_GEN_ID_TESTVAL_UNPAIRED, video_name)]
      split['music_names'] = [
          video_name for video_name in split['video_names']]
        
      # split['music_names'] = [
      #     get_testval_music_id(video_name) for video_name in split['video_names']]

  elif task == 'generation' and subset == 'train':
    split['video_names'] = [
        video_name for video_name in video_names
        if not re.match(MOTION_GEN_ID_TESTVAL, video_name) and \
           not re.match(MUSIC_ID_TESTVAL, video_name.split('_')[-2])]
    split['music_names'] = [
          video_name for video_name in split['video_names']]
          
    # split['music_names'] = [
    #     video_name.split('_')[-2] for video_name in split['video_names']]

  else:
    raise NotImplementedError

  return split



if __name__ == '__main__':
  data_dir = 'D:\\liushuyan12\\实验\\AIST\\data\\annotation_aist_plusplus_final\\motions\\'
  video_names = os.listdir(data_dir)
  video_names = sorted([name[:-4] for name in video_names])

  task, subset = 'prediction', 'train'
  split = get_split(video_names, task=task, subset=subset)
  print (task, subset, len(split['video_names']), len(split['music_names']))

  task, subset = 'prediction', 'val'
  split = get_split(video_names, task=task, subset=subset)
  print (task, subset, len(split['video_names']), len(split['music_names']))

  task, subset = 'prediction', 'test'
  split = get_split(video_names, task=task, subset=subset)
  print (task, subset, len(split['video_names']), len(split['music_names']))

  task, subset = 'generation', 'train'
  split = get_split(video_names, task=task, subset=subset)
  print (task, subset, len(split['video_names']), len(split['music_names']))

  task, subset = 'generation', 'val'
  split = get_split(video_names, task=task, subset=subset, is_paired=True, rng_seed=42)
  print (task, subset, len(split['video_names']), len(split['music_names']))

  task, subset = 'generation', 'val'
  split = get_split(video_names, task=task, subset=subset, is_paired=False, rng_seed=42)
  print (task, subset, len(split['video_names']), len(split['music_names']))

  task, subset = 'generation', 'test'
  split = get_split(video_names, task=task, subset=subset, is_paired=True, rng_seed=58)
  print (task, subset, len(split['video_names']), len(split['music_names']))

  task, subset = 'generation', 'test'
  split = get_split(video_names, task=task, subset=subset, is_paired=False, rng_seed=58)
  print (task, subset, len(split['video_names']), len(split['music_names']))

  # for name1, name2 in zip(split['video_names'], split['music_names']):
  #   print (name1, name2)
