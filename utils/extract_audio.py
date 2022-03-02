from moviepy.editor import *
from tqdm import tqdm

filepath = "D:\liushuyan12\实验\AIST\data\c01"
audiopath = "D:\liushuyan12\实验\AIST\data\ext_audio"

files = os.listdir(filepath)
already = os.listdir(audiopath)
print('*************already', already)

for file in tqdm(files):
    name, ext = os.path.splitext(file)
    if name+'.wav' in already:
        continue
    video = VideoFileClip(os.path.join(filepath, file))
    audio = video.audio
    audio.write_audiofile(os.path.join(audiopath, name + '.wav'))