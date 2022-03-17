# -*- coding: utf-8 -*-
import random
import os
import sys
import time
import copy
import json
import tqdm
import shutil
import argparse
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import _init_paths
from evaluation.metric2 import calculate_metrics

from evaluation.evalution_metric import *
from datasets.loader import get_music_name, get_tempo
from datasets.dataloader import AISTDataset, paired_collate_fn1, paired_collate_fn2
import datasets.test_dataloader2 as td
from datasets.dataset_split_setting import get_split
from visualization import processing
from utils.get_network import get_network
from utils import fk
from utils.fk import *
from utils.rot9D_to3 import convert_to3d
from loss import LOSS
import models.smpl as smpl_model
from models.generator import Generator
from classifier.classifier import Classifier
# from configs.configs import motion_dir, music_dir, video_list, train_batch_size, val_batch_size, \
#     leaning_rate, epochs, repeat_time_per_epoch

import warnings

genres = ['gBR', 'gHO', 'gJB', 'gJS', 'gKR', 'gLH', 'gLO', 'gMH', 'gPO', 'gWA']
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--config', default=None, help='config_file')
parser.add_argument('--resume', default=False, action='store_true')
parser.add_argument('--resume_from', type=str, default='./checkpoints/')
parser.add_argument('save_path', default=None, metavar='SAVE_PATH', type=str, help='Path to the saved models')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers for each data loader, default 4.')
parser.add_argument('network', default=None, metavar='NETWORK', type=str, help='the network you want to use')
parser.add_argument('saved_version', default=None, metavar='VERSION', type=str, help='the version of network saved')
parser.add_argument('loss_version', default=1, metavar='LOSS', type=str, help='the version of the loss, default is L1')
parser.add_argument('--device_ids', default='0, 1, 2, 3', type=str,
                    help='comma separated indices of GPU to use, e.g. 0,1 for using GPU_0 and GPU_1, default 0.')
parser.add_argument('--classifier_path',
                    default="/export/home/lg/huang/code/music2dance_condition/classifier/best_model_512_512dim_60frame_rnn",
                    type=str, help='classifier model path.')


class TTA(object):
    def __init__(self, args):
        self.args = args
        # motion_dir = self.args.config.motion_dir
        # music_dir = args.music_dir
        # video_list = args.video_list
        # train_batch_size = args.train_batch_size
        # val_batch_size = args.val_batch_size
        # leaning_rate = args.leaning_rate
        # epochs = args.epochs
        # repeat_time_per_epoch = args.repeat_time_per_epoch
        if torch.cuda.is_available():
            print("GPU CUDA is available!")
            torch.cuda.manual_seed(1000)
            torch.cuda.manual_seed_all(1000)
        else:
            print("CUDA is not available! CPU is available!")
            torch.manual_seed(1000)
            random.seed(1000)

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        with open(self.args.config.video_list, 'r') as f:
            video_names = [l.strip() for l in f.readlines()]

        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids
        num_GPU = len(args.device_ids.split(','))
        num_workers = args.num_workers * num_GPU

        train_split = get_split(video_names, task='prediction', subset='train')
        train_video_names = train_split['video_names']
        train_music_names = train_split['music_names']

        val_split = get_split(video_names, task='prediction', subset='val')
        val_video_names = val_split['video_names']
        val_music_names = val_split['music_names']

        test_split = get_split(video_names, task='generation', subset='test', is_paired=True)
        test_video_names = test_split['video_names']
        test_music_names = test_split['music_names']

        if self.args.network == "Dancerevolution_2Transformer":
            paired_collate_fn = paired_collate_fn1
        elif self.args.network == "2tcnlstm":
            paired_collate_fn = paired_collate_fn2
        elif self.args.network == "CrossModalTr":
            paired_collate_fn = paired_collate_fn1

        train_dataset = AISTDataset(train_video_names, train_music_names, self.args.config.motion_dir,
                                    self.args.config.music_dir, self.args.config)
        self.trainloader = data.DataLoader(train_dataset, self.args.config.train_batch_size,
                                           num_workers=num_workers, shuffle=True,
                                           collate_fn=paired_collate_fn, pin_memory=True)
        val_dataset = AISTDataset(val_video_names, val_music_names, self.args.config.motion_dir,
                                  self.args.config.music_dir, self.args.config, sample=200)
        self.valloader = data.DataLoader(val_dataset, self.args.config.val_batch_size, collate_fn=paired_collate_fn)

        test_dataset = td.AISTDataset(test_video_names, test_music_names, self.args.config.motion_dir,
                                      self.args.config.music_dir)
        self.test_loader = data.DataLoader(test_dataset, self.args.config.test_batch_size,
                                           collate_fn=td.paired_collate_fn1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_network(args, self.device)
        print('The MODEL is {}'.format(args.network))
        classifier = Classifier(input_size=72, emb_dim=512, hidden_size=512, num_classes=10)
        ck = torch.load(args.classifier_path, map_location=lambda storage, loc: storage)
        classifier.load_state_dict(ck['model'])
        self.classifier = classifier
        self.model = nn.DataParallel(model.to(self.device))

        if self.args.loss_version == '1':
            self.loss_funcs = nn.MSELoss()
        else:
            self.loss_funcs = LOSS(self.args.config)

        self.optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()),
                                           lr=self.args.config.leaning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                 [int(0.6 * self.args.config.epochs),
                                                                  int(0.95 * self.args.config.epochs)],
                                                                 0.1)

        self.best_val_loss = float("inf")
        self.best_beat_align = 0
        self.best_acc = 0
        self.best_fid = float('inf')
        self.best_div = 0
        self.best_fid_k = float('inf')
        self.best_fid_g = float('inf')
        self.best_dist_k = 0
        self.best_dist_g = 0
        self.best_model = None
        self.is_best = False
        self.last_epoch = 1
        self.metrics = {
            'beat_alignment': 0,
            'pred_acc': 0,
            'real_acc': 0,
            'div': 0,
            'fid': float('inf'),
            'fid_k': float("inf"),
            'fid_g': float("inf"),
            'dist_k': 0,
            'dist_g': 0,
        }

        self.summary_train = {'epoch': 0, 'step': 0}
        self.summary_valid = {'loss': float('inf')}
        self.summary_writer = SummaryWriter(args.save_path)

        self.val_dir = args.save_path + '/valid'
        self.train_dir = args.save_path + '/train'
        if not os.path.exists(self.val_dir):
            os.makedirs(self.val_dir)
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        self.target_genTime = False
        if self.args.resume:
            ck = torch.load(self.args.resume_from, map_location=lambda storage, loc: storage)
            self.best_val_loss = ck['best_val_loss']
            # self.best_pos_fd = ck['metric']['pos_fd']
            self.best_beat_align = ck['metric']['beat_alignment']
            # self.best_vel_fd = ck['metric']['vel_fd']
            self.best_fid = ck['metric']['fid']
            self.best_acc = ck['metric']['pred_acc']
            self.best_div = ck['metric']['div']
            # self.best_beat_hit = ck['metric']['beat_hit']
            self.best_fid_k = ck['metric']['fid_k']
            self.best_fid_g = ck['metric']['fid_g']
            self.best_dist_k = ck['metric']['dist_k']
            self.best_dist_g = ck['metric']['dist_g']
            self.optimizer.load_state_dict(ck['optimizer'])
            self.lr_scheduler.load_state_dict(ck['lr_scheduler'])
            self.model.load_state_dict(ck['state_dict'])
            self.last_epoch = ck['epoch']
            self.metrics = ck['metric']
            print('Successfully load resume model from: {}'.format(self.args.resume_from))

    def run(self):
        print('\nStart running.')
        for epoch in range(self.last_epoch, self.args.config.epochs + 1):
            # for epoch in range(25, epochs + 1):
            epoch_start_time = time.time()
            self.train_epoch(epoch)
            # val_loss, metrics = self.evaluate(epoch, self.args, self.classifier, self.device)
            self.lr_scheduler.step()

            if epoch % 100 == 0:
                val_loss, metrics = self.evaluate(epoch, self.args, self.classifier, self.device)
                save_checkpoint(self.args.network,
                                {  # 'epoch': epoch + 1,
                                    'state_dict': self.model.state_dict(),
                                    # 'val_loss': val_loss,
                                    # 'best_val_loss': self.best_val_loss,
                                    # 'optimizer': self.optimizer.state_dict(),
                                    # 'lr_scheduler': self.lr_scheduler.state_dict(),
                                    'metric': metrics,
                                }, False, self.args.save_path, epoch=epoch)
                # for k, v in metrics.items():
                #     print(f'{k}: {metrics[k]:.4f}')
                print('-' * 66)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f}'
                      .format(epoch, (time.time() - epoch_start_time), val_loss))
                print('-' * 66)

                print(self.summary_train['epoch'])
                self.summary_writer.add_scalar('valid/loss', val_loss, self.summary_train['epoch'])
            else:
                val_loss = self.best_val_loss
                metrics = self.metrics

            # if self.best_pos_fd > metrics['pos_fd'] \
            #         and self.best_vel_fd > metrics['vel_fd'] \
            if self.best_beat_align < metrics['beat_alignment'] \
                    and self.best_acc < metrics['pred_acc'] \
                    and self.best_fid > metrics['fid']:
                self.is_best = True
                self.best_val_loss = val_loss
                self.best_fid_k = metrics['fid_k']
                self.best_fid_g = metrics['fid_g']
                self.best_dist_k = metrics['dist_k']
                self.best_dist_g = metrics['dist_g']
                self.best_fid = metrics['fid']
                self.best_beat_align = metrics['beat_alignment']
                self.best_model = self.model
                self.metrics = metrics
                self.best_acc = metrics['pred_acc']
                self.best_div = metrics['div']
            else:
                self.is_best = False
            save_checkpoint(self.args.network,
                            {'epoch': epoch + 1,
                             'state_dict': self.model.state_dict(),
                             'val_loss': val_loss,
                             'best_val_loss': self.best_val_loss,
                             'optimizer': self.optimizer.state_dict(),
                             'lr_scheduler': self.lr_scheduler.state_dict(),
                             'metric': metrics,
                             }, self.is_best, self.args.save_path)

    def train_epoch(self, epoch):
        self.model.train()
        loss_items_name = ['sum', 'rotation', 'translation', '3D position', 'joint velocity', 'gram', 'genre']
        loss_items_cacher = [[], [], [], [], [], [], []]
        start_time = time.time()
        batch_inds = 0

        print('training epoch {}'.format(epoch))

        for _ in range(self.args.config.repeat_time_per_epoch):
            for batch_i, batch in enumerate(self.trainloader):

                if self.args.network == "Dancerevolution_2Transformer" or self.args.network == "CrossModalTr":
                    audio_seq, condition_motion, tgt_seq, video_name = batch
                    audio_seq, condition_motion, tgt_seq = [x.to(self.device) for x in audio_seq], [x.to(self.device)
                                                                                                    for x in
                                                                                                    condition_motion], tgt_seq.to(
                        self.device)
                elif self.args.network == "2tcnlstm":
                    audio_seq, condition_motion, tgt_seq = map(lambda x: x.to(self.device), batch[:3])
                    video_name = batch[3]

                if self.args.network != 'CrossModalTr':
                    gold_seq = tgt_seq[:, 1:]
                    tgt_seq = tgt_seq[:, :-1]
                    hidden = self.model.module.init_decoder_hidden(tgt_seq.size(0))
                    out_frame = torch.zeros(tgt_seq[:, 0].shape).to(self.device)  # (b, 72)
                    # out_frame = tgt_seq[:, 0]
                    out_seq = torch.FloatTensor(tgt_seq.size(0), 1).to(self.device)
                else:
                    gold_seq = tgt_seq
                    # tgt_seq = tgt_seq[:, :-1]
                    hidden = None
                    out_frame = None  # (b, 72)
                    # out_frame = tgt_seq[:, 0]
                    out_seq = None
                genre_label = np.array([genres.index(x[:3]) for x in video_name])
                genre_label = torch.LongTensor(genre_label)

                self.optimizer.zero_grad()
                if self.args.saved_version in ['11', '12', '13', '14']:
                    genre_onehot = torch.nn.functional.one_hot(genre_label, len(genres)).float()
                    genre_onehot = genre_onehot.to(self.device)
                    output = self.model(audio_seq, condition_motion, tgt_seq, regressive_len=20, type='full',
                                        genre_onehot=genre_onehot)
                else:
                    output = self.model(audio_seq, condition_motion, tgt_seq, regressive_len=20, type='full')
                genre_label = genre_label.to(self.device)

                if self.args.loss_version == '1':
                    min_len = min(output[0].shape[1], tgt_seq.shape[1])
                    tgt_seq = tgt_seq[:, :min_len, :]
                    loss = self.loss_funcs(output[0], tgt_seq)
                    loss_tuple = (loss, loss, loss, loss, loss, loss, loss,)
                elif self.args.saved_version in ['11', '12', '13', '14']:
                    loss_tuple = self.loss_funcs._calc_loss_(output, gold_seq, genre_label=None)
                else:
                    loss_tuple = self.loss_funcs._calc_loss_(output, gold_seq, genre_label=genre_label)

                loss_tuple[0].backward()

                self.optimizer.step()
                if epoch % 100 == 0:
                    if batch_i == len(self.trainloader) - 1:
                        pred, gt = convert_to3d(output[0], gold_seq)
                        for idx in tqdm.tqdm(range(0, self.args.config.train_batch_size)):
                            pred_poses = pred['pose'][idx].detach().cpu().numpy()
                            pred_trans = pred['trans'][idx].detach().cpu().numpy()

                            gt_poses = gt['pose'][idx].detach().cpu().numpy()
                            gt_trans = gt['trans'][idx].detach().cpu().numpy()

                            """save as gif"""
                            AISTDataset.plot_smpl_poses(pred_poses, pred_trans, save_type='pred',
                                                        fname='{}'.format(
                                                            batch_i * self.args.config.train_batch_size + idx),
                                                        test=False, test_dir=self.train_dir, epoch=epoch)
                            AISTDataset.plot_smpl_poses(gt_poses, gt_trans, save_type='targets',
                                                        fname='{}'.format(
                                                            batch_i * self.args.config.train_batch_size + idx),
                                                        test=False, test_dir=self.train_dir, epoch=epoch)

                            if idx == 1:
                                break

                for loss_inds, loss_item in enumerate(loss_tuple):
                    # print(loss_inds)
                    loss_items_cacher[loss_inds].append(loss_item.item())

                log_interval = 50
                if batch_i % log_interval == 0:
                    elapsed = time.time() - start_time
                    loss_log = ''
                    for loss_name, loss_cache in zip(loss_items_name, loss_items_cacher):
                        if len(loss_cache) > 0:
                            loss_log += '{}: {:.4f}  '.format(loss_name, np.array(loss_cache).mean())

                    print('| epoch {:2d} | {}/{} batches | ms/batch {:.2f} | {}'
                        .format(
                        epoch,
                        batch_i, len(self.trainloader) * self.args.config.repeat_time_per_epoch,
                                 elapsed * 1000 / log_interval,
                        loss_log
                    )
                    )
                    self.summary_writer.add_scalar('train/loss', np.array(loss_items_cacher[0]).mean(),
                                                   self.summary_train['step'])

                    loss_items_cacher = [[], [], [], [], [], [], []]
                    start_time = time.time()

                self.summary_train['step'] += 1
        self.summary_train['epoch'] += 1

        # if epoch%2==0:
        #     checkpoint = {'state_dict': self.model.state_dict(),'epoch': epoch}
        #     filename = os.path.join(self.args.save_path, f'epoch_{epoch}.pt')
        #     torch.save(checkpoint, filename)

    def evaluate(self, epoch, args, classifier, device, save_results=True):
        # self.model.eval()
        generator = Generator(args, epoch, device, model=self.model)
        smpl = smpl_model.create(model_path=self.args.config.smpl_model_path, gender='neutral', batch_size=1)

        print('\n==> Start Validating.')
        total_loss = []
        predict_joints = []
        target_joints = []
        music_paths = []
        genres_label = []
        bpm_list = []
        results = {}

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm.tqdm(self.test_loader, desc='Generating dance poses')):

                if self.args.network == "Dancerevolution_2Transformer" or self.args.network == "CrossModalTr":
                    audio_seq, condition_motion, tgt_seq = map(lambda x: x.to(device), batch[:3])
                    # audio_seq, condition_motion, tgt_seq, video_name = batch
                    # audio_seq, condition_motion, tgt_seq = [x.to(self.device) for x in audio_seq], [x.to(self.device)
                    #                                                                                 for x in
                    #                                                                                 condition_motion], tgt_seq.to(
                    #     self.device)
                elif self.args.network == "2tcnlstm":
                    audio_seq, condition_motion, tgt_seq = map(lambda x: x.to(self.device), batch[:3])
                video_name = batch[3]
                if self.args.network != 'CrossModalTr':
                    tgt_start = tgt_seq[:, :10, :]
                    poses = generator.generate(audio_seq, condition_motion, tgt_start)
                else:
                    self.model.eval()
                    if self.args.saved_version in ['11', '12', '13', '14']:
                        genre_label = np.array([genres.index(x[:3]) for x in video_name])
                        genre_label = torch.LongTensor(genre_label)
                        genre_onehot = torch.nn.functional.one_hot(genre_label, len(genres)).float()
                        genre_onehot = genre_onehot.to(self.device)
                        poses = self.model.module.generate(audio_seq, condition_motion, tgt_seq, regressive_len=60,
                                                           target_len=20 * 60, type='full',
                                                           genre_onehot=genre_onehot)
                    elif self.args.saved_version == '0':
                        poses = self.model.module.generate(audio_seq, condition_motion, tgt_seq, regressive_len=1,
                                                           target_len=20 * 60, type='full')
                    else:
                        poses = self.model.module.generate(audio_seq, condition_motion, tgt_seq, regressive_len=60,
                                                           target_len=20 * 60, type='full')

                pred, gt = convert_to3d(poses, tgt_seq)
                for idx in range(poses.shape[0]):
                    pred_poses = pred['pose'][idx].detach().cpu().numpy()
                    pred_trans = pred['trans'][idx].detach().cpu().numpy()
                    gt_poses = gt['pose'][idx].detach().cpu().numpy()
                    gt_trans = gt['trans'][idx].detach().cpu().numpy()
                    # print(video_name[idx])
                    # music_name = video_name[idx].split('_')[-3]
                    # music_tempo = get_tempo(music_name)
                    # audio_path = os.path.join(self.args.config.music_dir, video_name[idx] + '.wav')
                    # genre = genres.index(video_name[idx][:3])
                    music_name = get_music_name(video_name[idx])
                    music_tempo = get_tempo(music_name)
                    audio_path = os.path.join(self.args.config.test_music_dir, video_name[idx] + '.wav')
                    print('audio_path', audio_path)
                    genre = genres.index(video_name[idx][:3])
                    # print('audio_path', audio_path)

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

                    predict_joint = fk.SMPLForwardKinematics().from_aa(pred_poses, pred_trans)
                    target_joint = fk.SMPLForwardKinematics().from_aa(gt_poses, gt_trans)
                    # seq_len = predict_joint.shape[0]
                    music_paths.append(audio_path)
                    # predict_joints.append(predict_joint)
                    # target_joints.append(target_joint)
                    predict_joints.append(keypoints3d_pre)
                    target_joints.append(keypoints3d_tgt)
                    genres_label.append(genre)
                    bpm_list.append(music_tempo)

                # loss_tuple = nn.L1Loss()(poses[:, :tgt_seq.shape[1], :], tgt_seq)
                loss_tuple = torch.Tensor([0.])

                total_loss.append(loss_tuple.item())
                if batch_idx == len(self.test_loader) - 1:
                    # pred, gt = convert_to3d(poses, tgt_seq)
                    for idx in range(0, self.args.config.test_batch_size):
                        """tensorboard add_video"""
                        pred_poses = pred['pose'][idx].detach().cpu().numpy()
                        pred_trans = pred['trans'][idx].detach().cpu().numpy()
                        gt_poses = gt['pose'][idx].detach().cpu().numpy()
                        gt_trans = gt['trans'][idx].detach().cpu().numpy()

                        """save as gif"""
                        AISTDataset.plot_smpl_poses(pred_poses, pred_trans, save_type='pred',
                                                    fname='{}'.format(
                                                        batch_idx * self.args.config.test_batch_size + idx),
                                                    test=False, test_dir=self.val_dir, epoch=epoch)

                        AISTDataset.plot_smpl_poses(gt_poses, gt_trans, save_type='targets',
                                                    fname='{}'.format(
                                                        batch_idx * self.args.config.test_batch_size + idx),
                                                    test=False, test_dir=self.val_dir, epoch=epoch)
                        if idx > 2:
                            break
            genres_label = np.array(genres_label)
            metrics = calculate_metrics(music_paths, predict_joints, target_joints, bpm_list, classifier, genres_label,
                                        tol=12, sigma=3, start_id=0,
                                        verbose=True)
            results['metrics'] = metrics
            results['music_paths'] = music_paths
            results['predict_joints'] = predict_joints
            results['target_joints'] = target_joints
            if save_results:
                with open(os.path.join(self.val_dir, 'results.pkl'), 'wb') as f_save:
                    pickle.dump(results, f_save)
        self.target_genTime = True
        return np.array(total_loss).mean(), metrics


def save_checkpoint(network, state, is_best, checkpointdir, epoch=None):
    if epoch is None:
        filename = 'checkpoint.pth.tar'
    else:
        filename = 'checkpoint_' + str(epoch) + '.pth.tar'
    filepath = os.path.join(checkpointdir, filename)
    torch.save(state, filepath)
    if is_best:
        print('save best model!')
        shutil.copyfile(filepath, os.path.join(checkpointdir, 'model_best.pth.tar'))


def main():
    args = parser.parse_args()
    if args.network == 'Dancerevolution_2Transformer' or args.network == '2tcnlstm':
        from configs.configs_test import config
        args.config = config
    elif args.network == 'CrossModalTr':
        from configs.configs_train import config
        args.config = config
    tta = TTA(args)
    tta.run()


if __name__ == '__main__':
    main()
