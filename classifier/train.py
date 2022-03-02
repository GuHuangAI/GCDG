# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


""" This script handling the training process. """
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import random
import argparse
import sys

sys.path.append('/export2/home/lsy/music2dance_code')
import json
import numpy as np
import torch
import torch.nn as nn
import models.smpl as smpl_model
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset
import torch.nn.functional as F
from classifier.classifier import Classifier
from visualization import processing
from datasets import loader
from datasets.dataset_split_setting import get_split
from utils import fk
from utils.rot9D_to3 import single_joint_convert_to3d
from configs.configs import video_list, motion_dir, music_dir, ignore_list, sequence_list, MF_CACHE_DIR, \
    train_batch_size

SR = 60 * 512


def load_data(data_dir, interval):
    dance_data, labels = [], []
    fnames = sorted(os.listdir(data_dir))
    if ".ipynb_checkpoints" in fnames:
        fnames.remove(".ipynb_checkpoints")
    # fnames = fnames[:60]  # For debug
    for fname in fnames:
        path = os.path.join(data_dir, fname)
        label = -1
        if "hiphop" in fname:
            label = 0
        elif "ballet" in fname:
            label = 1
        elif "manako" in fname:
            label = 2

        with open(path) as f:
            sample_dict = json.loads(f.read())
            np_dance = np.array(sample_dict['dance_array'])
            # Only use 25 keypoints skeleton (basic bone) for 2D
            np_dance = np_dance[:, :50]

            seq_len, dim = np_dance.shape
            for i in range(0, seq_len, interval):
                dance_sub_seq = np_dance[i: i + interval]
                if len(dance_sub_seq) == interval and label != -1:
                    labels.append(label)
                    dance_data.append(dance_sub_seq)

    return dance_data, labels


class AISTDataset(Dataset):
    """A dataset class for loading, processing and plotting AIST++"""

    def __init__(self, motion_list, music_list, motion_dir, music_dir):
        self.motion_list = motion_list
        self.music_list = music_list
        self.motion_dir = motion_dir
        self.music_dir = music_dir
        self.video_names = [l.strip() for l in self.motion_list]

        with open(ignore_list, 'r') as f:
            ignore_video_names = [l.strip() for l in f.readlines()]
            self.video_names = [name for name in self.video_names if name not in ignore_video_names]

        with open(sequence_list, 'r') as f:
            all_sequence_names = [l.strip().split('\t')[0] for l in f.readlines()]

        self.sequence_names = [name for name in all_sequence_names if
                               name[:25] in self.video_names]  # train 10205; val 240; test 1110
        self.sequence_names = sorted(self.sequence_names)

        self.smpl_model = smpl_model.create(model_path='.', gender='neutral', batch_size=1)
        self.fk_engine = fk.SMPLForwardKinematics()
        self.genres = ['gBR', 'gHO', 'gJB', 'gJS', 'gKR', 'gLH', 'gLO', 'gMH', 'gPO', 'gWA']
        self.data = self.get_data(self.sequence_names)
        print(len(self.data))

    def __len__(self):
        # return len(self.sequence_names)
        return len(self.data)

    def __getitem__(self, index):
        #sequence_name = self.sequence_names[index]
        #return self.get_item(sequence_name)
        # data = self.get_data(self.sequence_names)
        return self.data[index]
    
    def get_data(self, sequence_names):
        data = []
        for sequence_name in sequence_names:
            joint, label, _ = self.get_item(sequence_name)
            for k in range(joint.shape[0]//60):
                joint_tmp = joint[k*60:(k+1)*60, :]
                data.append((joint_tmp, label, label))
        return data

    def get_item(self, video_name, verbose=True):
        seq_name, view = loader.get_seq_name(video_name[:25])
        frame = int(video_name[:-4].split('_')[-1]) * 60
        
        """motion"""
        choose_motion = {}
        motion_path = os.path.join(self.motion_dir, f'{seq_name}.pkl')
        motion_data = loader.load_pkl_lsy(motion_path, keys=['smpl_poses', 'smpl_scaling', 'smpl_trans'])
        motion_data['smpl_trans'] /= motion_data['smpl_scaling']
        del motion_data['smpl_scaling']
        del motion_data['smpl_loss']

        for k, v in motion_data.items():
            choose_motion[k] = motion_data[k][frame:frame + 240]

        smpl_poses = np.zeros((240, 24, 9))
        choose_motion['smpl_poses'] = choose_motion['smpl_poses'].reshape(-1, 24, 3)
        for i in range(240):
            smpl_poses[i,] = np.apply_along_axis(processing.transform_rot_representation, 1,
                                                 choose_motion['smpl_poses'][i,]).reshape(24, 9)
        smpl_poses = smpl_poses.reshape(-1, 216)

        final_motion = np.column_stack((smpl_poses, choose_motion['smpl_trans']))

        print(final_motion.shape)
        pred_poses = torch.from_numpy(final_motion[:, :216])
        pred_trans = final_motion[:, 216:]
        rot9d = pred_poses.reshape(-1, 24, 9).reshape(-1, 24, 3, 3).reshape(-1, 3, 3)
        pred_pose = rotation_matrix_to_angle_axis(rot9d).reshape(-1, 24 * 3)
        joints = self.smpl_model.forward(
            global_orient=pred_pose[:, :3].float(),
            body_pose=pred_pose[:, 3:].float(),
            transl=torch.from_numpy(pred_trans).float(),
            betas=torch.zeros(pred_pose.shape[0], 10,
                              device=pred_pose[:, :3].device).float(),
            return_verts=True, return_full_pose=True
        ).joints.detach().numpy()
        print(joints.shape)

        tmp = torch.from_numpy(final_motion).unsqueeze(0)
        tmp = single_joint_convert_to3d(tmp)
        poses = tmp['pose'][0].numpy()
        trans = tmp['trans'][0].numpy()

        final_joint = fk.SMPLForwardKinematics().from_aa(poses, trans)
        final_joint = final_joint.reshape((final_joint.shape[0], -1))
        label = self.genres.index(seq_name[:3])
        return final_joint, label, video_name[:-4]


class DanceDataset(Dataset):
    def __init__(self, dances, labels=None):
        if labels is not None:
            assert (len(labels) == len(dances)), \
                'the number of dances should be equal to the number of labels'
        self.dances = dances
        self.labels = labels

    def __len__(self):
        return len(self.dances)

    def __getitem__(self, index):
        if self.labels is not None:
            return self.dances[index], self.labels[index]
        else:
            return self.dances[index]


def save(model, optimizer, lr_schd, best_acc, save_dir, save_prefix, epoch):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    ck = {}
    save_path = os.path.join(save_dir, save_prefix)
    ck['model'] = model.state_dict()
    ck['optimizer'] = optimizer.state_dict()
    ck['lr_schd'] = lr_schd.state_dict()
    ck['best_acc'] = best_acc
    ck['last_epoch'] = epoch
    torch.save(ck, save_path)


def eval(dev_data, classifier, device, args):
    classifier.eval()
    corrects, avg_loss, size = 0, 0, 0
    for i, batch in enumerate(dev_data):
        dance, label, _ = batch
        dance = dance.to(device)
        label = label.to(device)
        # print(dance.shape)
        # dance, label, _ = map(lambda x: x.to(device), batch)
        dance = dance.type(torch.cuda.FloatTensor)
        logits, _ = classifier(dance)
        loss = F.cross_entropy(logits, label)
        avg_loss += loss.item()
        corrects += (torch.max(logits, 1)
                     [1].view(label.size()).data == label.data).sum()
        size += batch[0].shape[0]

    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy


def train(train_data, dev_data, classifier, device, args):
    optimizer = optim.Adam(filter(lambda x: x.requires_grad,
                                  classifier.parameters()), lr=args.lr)
    lr_schd = optim.lr_scheduler.MultiStepLR(optimizer, [int(0.6 * args.epochs), int(0.9 * args.epochs)], 0.1)
    best_acc = 0
    # steps = 0
    last_epch = 0
    if args.resume:
        ck = torch.load(args.resume_model, map_location=lambda storage, loc: storage)
        classifier.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        lr_schd.load_state_dict(ck['lr_schd'])
        best_acc = ck['best_acc']
        last_epch = ck['last_epoch']

    classifier.train()
    for epoch in range(1 + last_epch, args.epochs + 1):
        for i, batch in enumerate(train_data):
            # print(type(batch[0]))
            # print(type(batch[1]))
            # print(type(batch[2]))
            dance, label, _ = batch
            dance = dance.to(device)
            label = label.to(device)
            # print(label)
            # dance, label, _ = map(lambda x: x.to(device), batch)
            dance = dance.type(torch.cuda.FloatTensor)
            optimizer.zero_grad()
            logits, _ = classifier(dance)
            loss = F.cross_entropy(logits, label)
            loss.backward()
            optimizer.step()
            # steps += 1

            if i % args.log_interval == 0:
                corrects = (torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()
                train_acc = 100.0 * corrects / batch[0].shape[0]
                print('\rEpoch/Batch[{}/{}] - loss: {:.6f}  acc: {:.4f}%({}/{})  best_acc: {:.4f}'.format(epoch,
                                                                                        i,
                                                                                        loss.item(),
                                                                                        train_acc,
                                                                                        corrects,
                                                                                        batch[0].shape[0],
                                                                                        best_acc))
        lr_schd.step()
        # evaluate the model on test set at each epoch
        dev_acc = eval(dev_data, classifier, device, args)
        if dev_acc > best_acc:
            best_acc = dev_acc
            print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
            save(classifier, optimizer, lr_schd, best_acc, args.save_dir, args.save_model, epoch)


def prepare_dataloader(dance_data, labels, args):
    data_loader = torch.utils.data.DataLoader(
        DanceDataset(dance_data, labels),
        num_workers=8,
        batch_size=args.batch_size,
        shuffle=True,
        # collate_fn=paired_collate_fn,
        pin_memory=True
    )

    return data_loader


def main():
    """ Main function """
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir', type=str, default='data/train_1min',
                        help='the directory of dance data')
    parser.add_argument('--valid_dir', type=str, default='data/valid_1min',
                        help='the directory of music feature data')
    parser.add_argument('--save_model', type=str, default='best_model_512_512dim_60frame_rnn',
                        help='model name')
    parser.add_argument('--save_dir', metavar='PATH', default='classifier/')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--test_interval', type=int, default=50)
    parser.add_argument('--interval', type=int, default=50)
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--resume_model', type=str, default='./classifier/best_model_512_512dim_60frame_rnn')

    args = parser.parse_args()

    # Loading data
    with open(video_list, 'r') as f:
        video_names = [l.strip() for l in f.readlines()]

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids
    # num_GPU = len(args.device_ids.split(','))
    # num_workers = args.num_workers * num_GPU
    classifier = Classifier(input_size=72, emb_dim=512, hidden_size=512, num_classes=10)
    
    train_split = get_split(video_names, task='prediction', subset='train')
    train_video_names = train_split['video_names']
    train_music_names = train_split['music_names']

    val_split = get_split(video_names, task='prediction', subset='val')
    val_video_names = val_split['video_names']
    val_music_names = val_split['music_names']
    
    test_split = get_split(video_names, task='prediction', subset='test')
    test_video_names = test_split['video_names']
    test_music_names = test_split['music_names']

    train_dataset1 = AISTDataset(train_video_names, train_music_names, motion_dir, music_dir)
    train_dataset2 = AISTDataset(test_video_names, test_music_names, motion_dir, music_dir)
    train_dataset = torch.utils.data.ConcatDataset([train_dataset1, train_dataset2])
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True,
                                             pin_memory=True)
    val_dataset = AISTDataset(val_video_names, val_music_names, motion_dir, music_dir)
    dev_data = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False,
                                           pin_memory=True)
    # dance_data, labels = load_data(args.train_dir, args.interval)
    # z = list(zip(dance_data, labels))
    # random.shuffle(z)
    # dance_data, labels = zip(*z)

    # train_data = prepare_dataloader(dance_data[1000:], labels[1000:], args)
    # dev_data = prepare_dataloader(dance_data[:1000], labels[:1000], args)

    device = torch.device('cuda')

    # for name, parameters in classifier.named_parameters():
    #     print(name, ':', parameters.size())

    classifier = classifier.to(device)
    train(train_data, dev_data, classifier, device, args)


if __name__ == '__main__':
    main()
