# GENRE-CONDITIONED LONG-TERM 3D DANCE GENERATION DRIVEN BY MUSIC (ICASSP 2022)

## Network
<!-- [IMAGE] -->
<div align=center>
<img src="https://github.com/GuHuangAI/GCDG/releases/download/v1/framework.png" width="80%"/>
</div>

## Data Preparation
Download AIST++ from [url](https://google.github.io/aistplusplus_dataset/download.html). Run './utils/extrac_audio.py' to split the original audio to 240 seconds sequences, and run './utils/ext_audio_features_raw.py' to save the cache features.

## Run
<pre><code>
For training:
sh run_cmtr.sh

For testing:
Firstly, u should download the [wav.zip](), and use './utils/ext_audio_features_raw.py' to extract the cache features. And run:
sh run_test2.sh
</code></pre>
