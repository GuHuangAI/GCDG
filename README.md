# GENRE-CONDITIONED LONG-TERM 3D DANCE GENERATION DRIVEN BY MUSIC (ICASSP 2022)

## Table of contents
* [Abstract](#Abstract)
* [Network](#Network)
* [Data Preparation](#Data_Preparation)
* [Run](#Run)
* [Q&&A](#Q&&A)
* [Thanks](#Thanks)

## Abstract
Dancing to music is an artistic behavior of humans, however, letting machines generate dances from music is still challenging. Most existing works have been made progress in tackling the problem of motion prediction conditioned by music, yet they rarely consider the importance of the musical genre. In this paper, we focus on generating long-term 3D dance from music with a specific genre. Specifically, we construct a pure transformer-based architecture to correlate motion features and music features. To utilize the genre information, we propose to embed the genre categories into the transformer decoder so that it can guide every frame. Moreover, different from previous inference schemes, we introduce the motion queries to output the dance sequence in parallel that significantly improves the efficiency.

## Network
<!-- [IMAGE] -->
<div align=center>
<img src="https://github.com/GuHuangAI/GCDG/releases/download/v1/framework.png" width="80%"/>
</div>

## Data_Preparation
Download AIST++ from [url](https://google.github.io/aistplusplus_dataset/download.html). Run `./utils/extrac_audio.py` to split the original audio to 240 seconds sequences, and run `./utils/ext_audio_features_raw.py` to save the cache features.
The final data path follows as:
<pre><code>
├── AIST
│   ├── ext_audio (download from AIST)
│   ├── audio_sequence (split from ./ext_audio)
│   ├── motions (download from AIST)
│   ├── audio_sequence_features_raw (cache features extracted from ./audio_sequence)
│   ├── wav (download from AIST for testing)
│   ├── wav_features_aist (cache features extracted from ./wav)
      .
      .
      .
</code></pre>

## Run
For testing, u should download the [wav.zip](https://aistdancedb.ongaaccel.jp/v1.0.0/audio/wav.zip), and use `./utils/ext_audio_features_raw.py` to extract the cache features. Then, input correct path dirs in `./config/configs_train.py` and `./configs/configs_test.py`.

Notice that the **model v12** is used in the paper, and we present a new **model v13** now in which the motion query is obtained by using a linear layer to project the initial motion. You can change the model version in `run_cmtr.sh` and `run_test2.sh`.
<pre><code>
For training:
sh run_cmtr.sh

For testing, please download the model weights for **model v12**: https://pan.baidu.com/s/1fGA9INeAQA0FAMbmLxMQdg?pwd=2050
sh run_test2.sh
</code></pre>

Results:
<!-- [IMAGE] -->
<div align=center>
<img src="https://github.com/GuHuangAI/GCDG/releases/download/v1/result.png" width="80%"/>
</div>

## Q&A
If u have any questions, please concat with huangyuhang@shu.edu.cn.

## Thanks
Thanks to [Shuyan Liu](https://github.com/sy1019) for providing the base code and [Yu Sun](https://github.com/Arthur151) for smpl visualization.
