# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.
""" This module will handle the pose generation. """
import os
import torch
import torch.nn as nn
from models.TCN2_LSTM1 import TemporalConvNet, Decoder, Model
from models.dancerevolution_2Transformer import Encoder, Decoder, Model, get_subsequent_mask
import numpy as np
from utils.get_network import get_network
np.set_printoptions(threshold=np.inf)


class Generator(object):
    """ Load with trained model """
    def __init__(self, args, epoch, device, model=None):
        self.device = device
        if model:
            self.model = model
        else:
            print('==> Loading model..')
            checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint_{}.pth.tar'.format(epoch)))
            model = get_network(args, self.device)
            model = nn.DataParallel(model.to(self.device))
            model.load_state_dict(checkpoint['state_dict'])
        # model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()})
            print('[Info] Trained model.')

        self.args = args
        self.model = model.to(self.device)
        self.model.eval()

    def generate(self, audio_seq, condition_motion, tgt_seq, total_len=120):
        """ Generate dance pose in one batch """
        with torch.no_grad():
            if isinstance(condition_motion, list):
                src_seq_len = condition_motion[0].size(1)
            else:
                src_seq_len = condition_motion.size(1)

            bsz, tgt_seq_len, dim = tgt_seq.size()
            generated_frames_num = total_len - tgt_seq_len

            hidden = self.model.module.init_decoder_hidden(tgt_seq.size(0))
            out_frame = torch.zeros(tgt_seq[:, 0].shape).to(self.device)
            out_seq = torch.FloatTensor(tgt_seq.size(0), 1).to(self.device)
            vec_h, vec_c = hidden

            if isinstance(audio_seq, list):
                src_seq = audio_seq[0][:, :-1]
                src_pos1 = audio_seq[1][:, :-1]
                enc_mask1 = get_subsequent_mask(src_seq, 100)
                enc_outputs1, *_ = self.model.module.encoder1(src_seq, src_pos1, mask=enc_mask1)

                '''motion & embedding'''
                condition_seq = condition_motion[0][:, :-1]
                src_pos2 = condition_motion[1][:, :-1]
                enc_mask2 = get_subsequent_mask(condition_seq, 100)
                enc_outputs2, *_ = self.model.module.encoder2(condition_seq, src_pos2, mask=enc_mask2)

            else:
                audio_seq = audio_seq[:, :-1]
                condition_motion = condition_motion[:, :-1]
                enc_outputs1 = self.model.module.encoder1(audio_seq)
                enc_outputs2 = self.model.module.encoder2(condition_motion)

            enc_outputs = torch.cat([enc_outputs1, enc_outputs2], 1) # (b, 358, 200)

            for i in range(tgt_seq_len):
                dec_input = tgt_seq[:, i]
                dec_output, vec_h, vec_c = self.model.module.decoder(dec_input, vec_h, vec_c)
                dec_output = torch.cat([dec_output, enc_outputs[:, i]], 1)
                dec_output = self.model.module.linear(dec_output)
                out_seq = torch.cat([out_seq, dec_output], 1)

            for i in range(generated_frames_num):
                dec_input = dec_output
                dec_output, vec_h, vec_c = self.model.module.decoder(dec_input, vec_h, vec_c)
                dec_output = torch.cat([dec_output, enc_outputs[:, i + tgt_seq_len]], 1)
                dec_output = self.model.module.linear(dec_output)
                out_seq = torch.cat([out_seq, dec_output], 1)

            out_seq = out_seq[:, 1:].view(bsz, -1, dim)

        return out_seq