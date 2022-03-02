"""Library for image and video processing using ffmpeg."""

import os


def save_to_movie(out_path, frame_path_format, fps=30, start_frame=0):
  """Creates an mp4 video clip by using already stored frames.

  Args:
    out_path: The video path to store.
    frame_path_format: The image frames format. e.g., <frames_path>/%05d.jpg
    fps: FPS for the output video.
    start_frame: The start frame index.
  """
  # create movie and save it to destination
  command = [
      'ffmpeg',
      '-start_number',
      str(start_frame),
      '-framerate',
      str(fps),
      '-r',
      str(fps),  # output is 30 fps
      '-loglevel',
      'panic',
      '-i',
      frame_path_format,
      '-c:v',
      'libx264',
      '-preset',
      'slow',
      '-profile:v',
      'high',
      '-level:v',
      '4.0',
      '-pix_fmt',
      'yuv420p',
      '-y',
      out_path
  ]
  os.system(' '.join(command))


def attach_audio_to_movie(video_path, audio_path, out_path):
  """Attach audio(wav) to video(mp4)."""
  command = [
      'ffmpeg', '-i',
      str(video_path), '-strict', '-2', '-i', audio_path, '-c:v', 'copy',
      '-strict', '-2', '-c:a', 'aac', out_path, '-shortest'
  ]
  os.system(' '.join(command))


def hstack_movies(video_path1, video_path2, out_path):
  """Stack two same-height videos horizontally."""
  command = [
      'ffmpeg', '-i',
      str(video_path1), '-i',
      str(video_path2), '-filter_complex', 'hstack', out_path
  ]
  os.system(' '.join(command))


def stack_movies3x3(video_paths, out_path):
  """Stack 9 same-size videos into a 3x3 grid video."""
  command = ['ffmpeg']
  for video_path in video_paths:
    command += ['-i', str(video_path)]
  command += ['-filter_complex',
              '"[0:v] setpts=PTS-STARTPTS, scale=qvga [a0]; ' +\
              '[1:v] setpts=PTS-STARTPTS, scale=qvga [a1]; ' +\
              '[2:v] setpts=PTS-STARTPTS, scale=qvga [a2]; ' +\
              '[3:v] setpts=PTS-STARTPTS, scale=qvga [a3]; ' +\
              '[4:v] setpts=PTS-STARTPTS, scale=qvga [a4]; ' +\
              '[5:v] setpts=PTS-STARTPTS, scale=qvga [a5]; ' +\
              '[6:v] setpts=PTS-STARTPTS, scale=qvga [a6]; ' +\
              '[7:v] setpts=PTS-STARTPTS, scale=qvga [a7]; ' +\
              '[8:v] setpts=PTS-STARTPTS, scale=qvga [a8]; ' +\
              '[a0][a1][a2][a3][a4][a5][a6][a7][a8]xstack=inputs=9:'+\
              'layout=0_0|w0_0|w0+w1_0|0_h0|w0_h0|w0+w1_h0|0_h0+h1|w0_h0+h1|w0+w1_h0+h1[out]"',
              '-map', '"[out]"', str(out_path)]
  os.system(' '.join(command))
