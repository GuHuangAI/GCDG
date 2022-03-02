import os
import matplotlib.pyplot as plt

from . import ffmpeg


def vis_music_motion(music_onset_strength=None, music_beats=None, music_xaxis=None,
    motion_onset_strength=None, motion_beats=None, motion_xaxis=None, out_dir=None,
    fname=None, current_xaxis=None, figsize=(19.2, 10.8)):
    
    show_music = not bool(music_onset_strength is None and music_beats is None)
    show_motion = not bool(motion_onset_strength is None and motion_beats is None)
    nrows = int(show_music) + int(show_motion)
    if nrows == 0:
        return

    fig, ax = plt.subplots(nrows=nrows, figsize=figsize)
    if nrows == 1:
        ax = [ax]

    if show_music:
        if music_onset_strength is not None:
            ax[0].plot(
                music_xaxis,
                music_onset_strength,
                label='Music Onset Strength')
        if music_beats is not None:
            ax[0].vlines(
                music_xaxis[music_beats],
                0,
                music_onset_strength.max(),
                color='r', alpha=0.5, linestyle='--', label='Music Beats')
        ax[0].legend(loc='upper left')

    if show_motion:
        if motion_onset_strength is not None:
            ax[1].plot(
                motion_xaxis,
                motion_onset_strength,
                label='Motion Onset Strength')
        if motion_beats is not None:
            ax[1].vlines(
                motion_xaxis[motion_beats],
                0,
                motion_onset_strength.max(),
                color='g', alpha=0.7, linestyle='--', label='Motion Beats')
        if music_beats is not None and current_xaxis is None:
            ax[1].vlines(
                music_xaxis[music_beats],
                0,
                motion_onset_strength.max(),
                color='r', alpha=0.3, linestyle='--', label='Music Beats')
        ax[1].legend(loc='upper left')
        ax[1].set_xticks(motion_xaxis[motion_beats])

    if current_xaxis is not None:
        window = 200
        xstart = current_xaxis - window//2
        xend = current_xaxis + window//2
        if xstart < 0:
            xstart = 0
            xend = window
        elif xend > motion_xaxis.max():
            xstart = motion_xaxis.max() - window,
            xstart = xstart[0] if type(xstart) is tuple else xstart
            xend = motion_xaxis.max()

        ax[0].set_xlim(xstart, xend)
        ax[0].vlines(
            current_xaxis,
            0,
            music_onset_strength.max(),
            color='black', alpha=1.0, linestyle='solid')

        ax[1].set_xlim(xstart, xend)
        ax[1].vlines(
            current_xaxis,
            0,
            motion_onset_strength.max(),
            color='black', alpha=1.0, linestyle='solid')

    if out_dir is None:
        plt.show()  # interactive
    else:
        save_to = os.path.join(out_dir, "frames", fname + "_onset")

        if not os.path.exists(save_to):
            os.makedirs(save_to)

        # Save frames to disk.
        fig.savefig(
            os.path.join(save_to, 'frame_{:0>4}.{}'.format(current_xaxis, "jpg")))

    plt.close()


def vis_music_motion_movie(
    music_onset_strength=None,
    music_beats=None,
    music_xaxis=None,
    motion_onset_strength=None,
    motion_beats=None,
    motion_xaxis=None,
    out_dir=None,
    fname=None,
    figsize=(19.2, 10.8),
    num_frames=None,
    to_video=True,
    audio_path=None,
    fps=30,
):
    for current_fs in range(num_frames):
        vis_music_motion(
            music_onset_strength, music_beats, music_xaxis,
            motion_onset_strength, motion_beats, motion_xaxis,
            out_dir=out_dir,
            fname=fname,
            current_xaxis=current_fs,
        )

    video_dir = os.path.join(out_dir, "videos")
    save_to = os.path.join(out_dir, "frames", fname + "_onset")

    if not os.path.exists(save_to):
        os.makedirs(save_to)

    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    # Create a video clip.
    if to_video:
        video_path = os.path.join(video_dir, fname + "_onset.mp4")
        ffmpeg.save_to_movie(
            video_path,
            os.path.join(save_to, 'frame_%04d.jpg'), fps=fps)

    animation_path = os.path.join(video_dir, fname + "_skeleton.mp4")
    if os.path.exists(animation_path):
        combine_path = os.path.join(video_dir, fname + "_combine.mp4")
        ffmpeg.hstack_movies(animation_path, video_path, combine_path)
    else:
        combine_path = video_path

    if audio_path:
        ffmpeg.attach_audio_to_movie(
            combine_path,
            audio_path,
            os.path.join(video_dir, fname + "_onset_audio.mp4")
        )


if __name__ == '__main__':
    import argparse
    import numpy as np
    import aist_plusplus_lib as aist
    from aist_plusplus_lib.beats import (
        get_envelope_music, get_envelope_motion,
        get_beats, get_peaks, get_local_min)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='D:\\liushuyan12\\experiments\\AIST\\data\\c01',
                        help='AIST video store directory.')
    parser.add_argument('--video_name', type=str,
                        default='gBR_sBM_c01_d04_mBR0_ch01',
                        help='AIST video name.')
    parser.add_argument('--pred_file', type=str,
                        default=None,
                        help='Prediction from ST-Transformer. (.npy)')
    args = parser.parse_args()

    # available_colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']
    music_path = os.path.join(args.data_dir, f'{args.video_name}.wav')
    motion_path = os.path.join(args.data_dir, f'{args.video_name}.pkl')

    if args.pred_file is None:
        data = aist.load(motion_path)
        joints = data['smpl_joints']
    else:
        joints = np.load(args.pred_file)

    envelope_mu, sr_mu, hop_length_mu, envelope_ts_mu = get_envelope_music(music_path)
    envelope_mo, sr_mo, hop_length_mo, envelope_ts_mo = get_envelope_motion(joints)
    music_beats = get_peaks(envelope_mu, sr_mu, hop_length_mu, envelope_ts_mu)
    motion_beats = get_local_min(envelope_mo, sr_mo, hop_length_mo, envelope_ts_mo)

    music_beats_inds = (music_beats * sr_mu / hop_length_mu + 0.5).astype(np.int32)
    motion_beats_inds = (motion_beats * sr_mo / hop_length_mo + 0.5).astype(np.int32)
    envelope_fs_mu = envelope_ts_mu * 30
    envelope_fs_mo = envelope_ts_mo * 30

    vis_music_motion_movie(
        envelope_mu, music_beats_inds, envelope_fs_mu,
        envelope_mo, motion_beats_inds, envelope_fs_mo,
        out_dir='./data/vis_plot/',
        fname=args.video_name,
        num_frames=joints.shape[0],
        to_video=True,
        audio_path=music_path,
    )
