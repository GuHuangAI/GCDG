# -*- coding: utf-8 -*-
import os
import tqdm
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import _init_paths
from datasets.loader import get_music_name, get_tempo
from datasets.test_dataloader2 import AISTDataset, paired_collate_fn1, paired_collate_fn2
from datasets.dataset_split_setting import get_split
import models.smpl as smpl_model
from utils import fk
from models.generator import Generator
from utils.get_network import get_network
from utils.rot9D_to3 import convert_to3d
from visualization.renderer import get_renderer, create_gif
# from evaluation.evalution_metric import motion_div, pos_frechet, vel_frechet
from evaluation.metric2 import calculate_metrics
from configs.configs_test import config

video_list, motion_dir, music_dir, ignore_list, sequence_list, MF_CACHE_DIR, test_batch_size, smpl_model_path = \
    config.video_list, config.motion_dir, config.test_music_dir, config.ignore_list, config.sequence_list, config.test_cache_dir, \
    config.test_batch_size, config.smpl_model_path
import warnings
from classifier.classifier import Classifier

warnings.filterwarnings("ignore")
genres = ['gBR', 'gHO', 'gJB', 'gJS', 'gKR', 'gLH', 'gLO', 'gMH', 'gPO', 'gWA']
parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('save_path', default=None, metavar='SAVE_PATH', type=str, help='Path to the saved models')
parser.add_argument('classifier_path', default=None, metavar='CLASSIFIER_PATH', type=str,
                    help='Path to the classifier models')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for each data loader, default 2.')
parser.add_argument('network', default=None, metavar='NETWORK', type=str, help='the network you want to use')
parser.add_argument('epoch', default=None, metavar='EPOCH', type=str, help='the network you want to use')
parser.add_argument('saved_version', default='1', metavar='VERSION', type=str, help='the network you want to use')
parser.add_argument('--regressive_len', default=60, type=int, help='')
parser.add_argument('--device_ids', default='0', type=str,
                    help='comma separated indices of GPU to use, e.g. 0,1 for using GPU_0 and GPU_1, default 0.')
parser.add_argument('--save_fig', default=False, action='store_true', help='')


def test(args, generator, testloader, device, test_dir, classifier, save_results=True, save_fig=False):
    total_loss = 0.
    epoch = args.epoch
    predict_joints = []
    target_joints = []
    music_paths = []
    genres_label = []
    bpm_list = []
    results = {}
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm.tqdm(testloader, desc='Generating dance poses')):
            audio_seq, condition_motion, tgt_seq = map(lambda x: x.to(device), batch[:3])
            # audio_seq, condition_motion, tgt_seq, video_name = batch
            # audio_seq, condition_motion, tgt_seq = [x.to(device) for x in audio_seq], \
            #                                       [x.to(device) for x in condition_motion], tgt_seq.to(device)
            video_name = batch[3]
            # print(video_name)
            # genre = np.array([genres.index(video[:3]) for video in video_name])
            # print(genre)
            tgt_start = tgt_seq[:, :10, :]
            # if batch_idx >9:
            #    continue
            if args.network == "2tcnlstm" or args.network == 'Dancerevolution_2Transformer':
                poses = generator.generate(audio_seq, condition_motion, tgt_start, total_len=20 * 60)
            else:
                if args.saved_version in ['11', '12', '13', '14']:
                    genre_label = np.array([genres.index(x[:3]) for x in video_name])
                    genre_label = torch.LongTensor(genre_label)
                    genre_onehot = torch.nn.functional.one_hot(genre_label, len(genres)).float()
                    genre_onehot = genre_onehot.to(device)
                    poses = generator.model.module.generate(audio_seq, condition_motion, tgt_seq, regressive_len=60,
                                                            target_len=20 * 60, type='full',
                                                            genre_onehot=genre_onehot)
                else:
                    poses = generator.model.module.generate(audio_seq, condition_motion, tgt_seq,
                                                            regressive_len=args.regressive_len, target_len=20 * 60,
                                                            type='full')
            pred, gt = convert_to3d(poses, tgt_seq)

            for idx in range(0, test_batch_size):
                pred_poses = pred['pose'][idx].detach().cpu().numpy()
                pred_trans = pred['trans'][idx].detach().cpu().numpy()
                gt_poses = gt['pose'][idx].detach().cpu().numpy()
                gt_trans = gt['trans'][idx].detach().cpu().numpy()
                # print(video_name[idx])
                music_name = get_music_name(video_name[idx])
                music_tempo = get_tempo(music_name)
                audio_path = os.path.join(music_dir, video_name[idx] + '.wav')
                print('audio_path', audio_path)
                genre = genres.index(video_name[idx][:3])
                smpl = smpl_model.create(model_path=smpl_model_path, gender='neutral', batch_size=1)

                # from smplx import SMPL
                # smpl = SMPL(model_path='./smpl', gender='neutral', batch_size=1)

                keypoints3d_pre = smpl.forward(
                    global_orient=torch.from_numpy(pred_poses[:, :3]).float(),
                    body_pose=torch.from_numpy(pred_poses[:, 3:]).float(),
                    transl=torch.from_numpy(pred_trans).float(),
                    betas=torch.zeros(pred_poses.shape[0], 10,
                                      device=torch.from_numpy(pred_poses[:, :3]).device).float(),
                    return_verts=True, return_full_pose=True
                ).joints.detach().numpy()[:, :24, :]
                keypoints3d_tgt = smpl.forward(
                    global_orient=torch.from_numpy(gt_poses[:, :3]).float(),
                    body_pose=torch.from_numpy(gt_poses[:, 3:]).float(),
                    transl=torch.from_numpy(gt_trans).float(),
                    betas=torch.zeros(gt_poses.shape[0], 10,
                                      device=torch.from_numpy(gt_poses[:, :3]).device).float(),
                    return_verts=True, return_full_pose=True
                ).joints.detach().numpy()[:, :24, :]
                # print('keypoints3d:', keypoints3d.shape)

                if save_fig:
                    '''
                    AISTDataset.plot_smpl_poses(pred_poses, pred_trans, save_type='pred',
                                                # fname='{}'.format(batch_idx * test_batch_size + idx),
                                                fname=video_name[idx],
                                                test=True, test_dir=test_dir, epoch=epoch, to_video=False,
                                                audio_path=None)
                    print('Pred gif Saved Over!')

                    AISTDataset.plot_smpl_poses(gt_poses, gt_trans, save_type='targets',
                                                # fname='{}'.format(batch_idx * test_batch_size + idx),
                                                fname=video_name[idx],
                                                test=True, test_dir=test_dir, epoch=epoch, to_video=False,
                                                audio_path=None)
                    print('GT gif Saved Over!')
                    '''
                    # smpl = smpl_model.create(model_path=smpl_model_path, gender='neutral', batch_size=1)
                    pred_vertices = smpl.forward(
                        global_orient=torch.from_numpy(pred_poses[:, :3]).float(),
                        body_pose=torch.from_numpy(pred_poses[:, 3:]).float(),
                        transl=torch.from_numpy(pred_trans).float(),
                        betas=torch.zeros(pred_poses.shape[0], 10,
                                          device=torch.from_numpy(pred_poses[:, :3]).device).float(),
                        return_verts=True, return_full_pose=True
                    ).vertices.detach().numpy()

                    # np.save(os.path.join(test_dir, 'pred_vertices/{}.npy'.format(batch_idx * test_batch_size + idx)), pred_vertices)
                    for i in tqdm.tqdm(range(0, pred_vertices.shape[0])):
                        get_renderer(smpl_model_path, pred_vertices[i], 'pred', test_dir, i, test=True,
                                     model_type='smpl')
                        # get_tex_renderer(smpl_model_path, pred_vertices[i], 'pred', test_dir, i, test=True, model_type='smpl')

                    # create_gif(os.path.join(test_dir, 'mesh_frames/pred'), os.path.join(test_dir, 'mesh/pred'),
                    #            batch_idx * test_batch_size + idx, 1. / 60, to_video=True, audio_path=audio_path)
                    create_gif(os.path.join(test_dir, 'mesh_frames/pred'), os.path.join(test_dir, 'mesh/pred'),
                               video_name[idx], 1. / 60, to_video=True, audio_path=audio_path)
                    print("Pred Mesh Saved Over!")

                    """targets:9d->3d"""
                    gt_vertices = smpl.forward(
                        global_orient=torch.from_numpy(gt_poses[:, :3]).float(),
                        body_pose=torch.from_numpy(gt_poses[:, 3:]).float(),
                        transl=torch.from_numpy(gt_trans).float(),
                        betas=torch.zeros(gt_poses.shape[0], 10,
                                          device=torch.from_numpy(gt_poses[:, :3]).device).float(),
                        return_verts=True, return_full_pose=True
                    ).vertices.detach().numpy()
                    # np.save(os.path.join(test_dir, 'gt_vertices/{}.npy'.format(batch_idx * test_batch_size + idx)), gt_vertices)
                    for i in tqdm.tqdm(range(0, gt_vertices.shape[0])):
                        get_renderer(smpl_model_path, gt_vertices[i], 'targets', test_dir, i, test=True,
                                     model_type='smpl')
                        # get_tex_renderer(smpl_model_path, gt_vertices[i], 'targets', test_dir, i, test=True, model_type='smpl')

                    # create_gif(os.path.join(test_dir, 'mesh_frames/targets'), os.path.join(test_dir, 'mesh/targets'),
                    #            batch_idx * test_batch_size + idx, 1. / 60)
                    create_gif(os.path.join(test_dir, 'mesh_frames/targets'), os.path.join(test_dir, 'mesh/targets'),
                               video_name[idx], 1. / 60)
                    print("GT Mesh Saved Over")

                predict_joint = fk.SMPLForwardKinematics().from_aa(pred_poses, pred_trans)
                # print('predict_joint', predict_joint[0])
                target_joint = fk.SMPLForwardKinematics().from_aa(gt_poses, gt_trans)
                # seq_len = predict_joint.shape[0]
                genres_label.append(genre)
                music_paths.append(audio_path)
                # predict_joints.append(predict_joint)
                # target_joints.append(target_joint)
                predict_joints.append(keypoints3d_pre)
                target_joints.append(keypoints3d_tgt)
                bpm_list.append(music_tempo)
        genres_label = np.array(genres_label)
        print(genres_label.shape)
        ### evaluation metrics ###
        print(test_dir)
        results['music_paths'] = music_paths
        results['predict_joints'] = predict_joints
        results['target_joints'] = target_joints
        metrics = calculate_metrics(music_paths, predict_joints, target_joints, bpm_list, classifier, genres_label,
                                    tol=12, sigma=3, start_id=0,
                                    verbose=True)
        results['metrics'] = metrics
        if save_results:
            with open(os.path.join(test_dir, 'results.pkl'), 'wb') as f_save:
                pickle.dump(results, f_save)


def run(args):
    if torch.cuda.is_available():
        print("GPU CUDAis available!")
        torch.cuda.manual_seed(1000)
        torch.cuda.manual_seed_all(1000)
    else:
        print("CUDA is not available! cpu is available!")
        torch.manual_seed(1000)

    with open(video_list, 'r') as f:
        video_names = [l.strip() for l in f.readlines()]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids

    test_split = get_split(video_names, task='generation', subset='test', is_paired=True)
    test_video_names = test_split['video_names']
    test_music_names = test_split['music_names']
    print('TEST data shape is: ', len(test_video_names))

    test_dir = args.save_path + '/test_' + str(args.regressive_len)
    print(test_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    # with open(config.test_video_list, 'r') as f:
    #     test_video_names = [l.strip() for l in f.readlines()]
    test_dataset = AISTDataset(test_video_names, test_music_names, motion_dir, music_dir)
    testloader = data.DataLoader(test_dataset, test_batch_size, collate_fn=paired_collate_fn1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = get_network(args)
    classifier = Classifier(input_size=72, emb_dim=512, hidden_size=512, num_classes=10)
    ck = torch.load(args.classifier_path, map_location=lambda storage, loc: storage)
    classifier.load_state_dict(ck['model'])
    print('The MODEL is {}'.format(args.network))
    generator = Generator(args, args.epoch, device)
    test(args, generator, testloader, device, test_dir, classifier, save_fig=args.save_fig)


def main():
    args = parser.parse_args()
    args.config = config
    run(args)


if __name__ == '__main__':
    main()
