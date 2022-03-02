# _DIR = '/export2/home/lsy/TTA_data'

from easydict import EasyDict as edict
config = edict()
# config.MF_CACHE_DIR = '/export/home/lg/data/TTA_data/audio_sequence_features_new'
config.MF_CACHE_DIR = '/export/home/lg/data/TTA_data/audio_sequence_features_raw'
config.motion_dir = "/export/home/lg/data/TTA_data/motions/"
config.music_dir = "/export/home/lg/data/TTA_data/audio_sequence/"
config.test_music_dir = "/export/home/lg/data/TTA_data/ext_audio/"
# config.test_cache_dir = "/export/home/lg/data/TTA_data/audio_features/"
#config.test_cache_dir = "/export/home/lg/data/TTA_data/wav_features_aist/"
config.test_cache_dir = "/export/home/lg/data/TTA_data/wav_features_dr/"
config.test_video_list = "/export/home/lg/data/TTA_data/longseq_40.txt"
# test_music_dir = "/export2/home/lsy/TTA_data/ext_audio/"
config.video_list = "/export/home/lg/data/TTA_data/video_list.txt"
config.ignore_list = "/export/home/lg/data/TTA_data/ignore_list.txt"
config.sequence_list = '/export/home/lg/data/TTA_data/audio_sequence.txt'
config.smpl_model_path = '.'

config.train_batch_size = 120
config.val_batch_size = 64
config.test_batch_size = 1

config.leaning_rate = 0.0001
config.epochs = 1500
config.repeat_time_per_epoch = 1

config.rotation_weight = 5.
config.trans_weight = 2.
config.pos_weight = 3. 
config.vel_weight = 2.
config.gram_weight = 0.007
