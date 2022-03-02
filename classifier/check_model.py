import torch
import os

model_paths = '/export2/home/lsy/music2dance_code/classifier/classifier'
for path in os.listdir(model_paths):
    model_path = os.path.join(model_paths, path)
    ck = torch.load(model_path, map_location='cpu')
    print('{}: {}'.format(path, ck['best_acc']))