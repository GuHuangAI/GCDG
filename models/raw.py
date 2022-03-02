""" Define the TCN-LSTM network """
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import models.smpl as smpl_model
from utils.rotation import rotation_matrix_to_angle_axis
from models.layers import MultiHeadAttention, PositionwiseFeedForward
from configs.configs import smpl_model_path, train_batch_size


def get_subsequent_mask(seq, sliding_windown_size):
    """ For masking out the subsequent info. """
    batch_size, seq_len, _ = seq.size()
    mask = torch.ones((seq_len, seq_len), device=seq.device, dtype=torch.uint8)
    mask = torch.triu(mask, diagonal=-sliding_windown_size)
    mask = torch.tril(mask, diagonal=sliding_windown_size)
    mask = 1 - mask
    mask = 1 - mask
    return mask.bool()


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, d_inner=1024, d_model=200, input_size=20, d_word_vec=10, kernel_size=2, dropout=0.1):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.src_emb = nn.Linear(input_size, d_word_vec)
        self.drop = nn.Dropout(dropout)
        self.d_model = d_model
        self.network = nn.Sequential(*layers)
        # self.decoder = nn.Linear(800, nout)
        # self.init_weights()

    # def init_weights(self):
    #     initrange = 0.1
    #     self.decoder.bias.data.fill_(0)
    #     self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):

        src = self.drop(self.src_emb(src))
        x = src.transpose(1,2)
        x = self.network(x)
        x = x.transpose(1,2)
        # print('TCN: ', x.shape)
        # x = self.decoder(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_size=274, d_word_vec=150, hidden_size=200,
                 dropout=0.1, encoder_d_model=200):
        super().__init__()

        self.input_size = input_size
        self.d_word_vec = d_word_vec
        self.hidden_size = hidden_size
      
        self.tgt_emb = nn.Linear(input_size, d_word_vec)
        self.dropout = nn.Dropout(dropout)
        self.encoder_d_model = encoder_d_model

        self.lstm1 = nn.LSTMCell(d_word_vec, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.lstm3 = nn.LSTMCell(hidden_size, hidden_size)

    def init_state(self, bsz, device):
        c0 = torch.randn(bsz, self.hidden_size).to(device)
        c1 = torch.randn(bsz, self.hidden_size).to(device)
        c2 = torch.randn(bsz, self.hidden_size).to(device)
        h0 = torch.randn(bsz, self.hidden_size).to(device)
        h1 = torch.randn(bsz, self.hidden_size).to(device)
        h2 = torch.randn(bsz, self.hidden_size).to(device)

        vec_h = [h0, h1, h2]
        vec_c = [c0, c1, c2]

        # ԭ��ʼ
        # BOP = BOS_POSE
        # BOP = np.tile(BOP, (bsz, 1))
        # root = BOP[:, 2*8:2*9]
        # bos = BOP - np.tile(root, (1, 25))
        # bos[:, 2*8:2*9] = root
        # out_frame = torch.from_numpy(bos).float().to(device)
        # out_seq = torch.FloatTensor(bsz, 1).to(device)

        return (vec_h, vec_c)#, out_frame, out_seq
        # return (vec_h, vec_c), out_frame, out_seq

    def forward(self, in_frame, vec_h, vec_c): 

        # print(in_frame.shape)
        in_frame = self.tgt_emb(in_frame)
        in_frame = self.dropout(in_frame)

        vec_h0, vec_c0 = self.lstm1(in_frame, (vec_h[0], vec_c[0]))
        vec_h1, vec_c1 = self.lstm2(vec_h[0], (vec_h[1], vec_c[1]))
        vec_h2, vec_c2 = self.lstm3(vec_h[1], (vec_h[2], vec_c[2]))

        vec_h_new = [vec_h0, vec_h1, vec_h2]
        vec_c_new = [vec_c0, vec_c1, vec_c2]
        return vec_h2, vec_h_new, vec_c_new


class Model(nn.Module):
    def __init__(self, encoder, decoder, condition_step=10,
                 sliding_windown_size=100, lambda_v=0.01, device=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.linear = nn.Linear(decoder.hidden_size + encoder.d_model, decoder.input_size)

        self.condition_step = condition_step
        self.sliding_windown_size = sliding_windown_size
        self.lambda_v = lambda_v
        self.device = device

        self.smpl_model = smpl_model.create(smpl_model_path, \
             batch_size=train_batch_size,model_type='smpl', gender='neutral', use_face_contour=False, ext='npz',flat_hand_mean=True, use_pca=False)


    def init_decoder_hidden(self, bsz):
        return self.decoder.init_state(bsz, self.device)

    # dynamic auto-condition + self-attention mask
    def forward(self, src_seq, src_pos, tgt_seq, gt, hidden, dec_output, out_seq, epoch_i):
        bsz, seq_len, _ = tgt_seq.size()
        vec_h, vec_c = hidden

        # enc_mask = get_subsequent_mask(src_seq, self.sliding_windown_size)
        enc_outputs = self.encoder(src_seq)

        groundtruth_mask = torch.ones(seq_len, self.condition_step)
        prediction_mask = torch.zeros(seq_len, int(epoch_i * self.lambda_v))
        mask = torch.cat([prediction_mask, groundtruth_mask], 1).view(-1)[:seq_len]  # for random

        for i in range(seq_len):
            dec_input = tgt_seq[:, i] if mask[i] == 1 else dec_output.detach()  # dec_output
            dec_output, vec_h, vec_c = self.decoder(dec_input, vec_h, vec_c)
            dec_output = torch.cat([dec_output, enc_outputs[:, i]], 1)
            dec_output = self.linear(dec_output)
            out_seq = torch.cat([out_seq, dec_output], 1)

        out_seq = out_seq[:, 1:].view(bsz, seq_len, -1)

        out_seq = self.post_process(out_seq, targets=gt)

        return out_seq

    def smpl_forward(self, pose):
        rot9d = pose.reshape(-1, 24, 9).reshape(-1, 24, 3, 3).reshape(-1, 3, 3)
        pred_pose = rotation_matrix_to_angle_axis(rot9d).reshape(-1, 24*3)
        params_dict = {'global_orient': pred_pose[:,:3], 'body_pose':pred_pose[:,3:], 'betas':torch.zeros(pred_pose.shape[0],10,device=rot9d.device).float()}
        smpl_out = self.smpl_model(**params_dict, return_verts=True, return_full_pose=True)

        return smpl_out

    def post_process(self, output, targets=None):
        output_pose = output[:, :, :216]
        pred_smpl_out = self.smpl_forward(output_pose)
        pred_joints = pred_smpl_out.joints[:,:24]
        out = [output, pred_joints]

        if targets is not None:
            targets = targets[:, :, :216]
            target_smpl_out = self.smpl_forward(targets)
            target_joints = target_smpl_out.joints[:,:24]
            out.append(target_joints)

        return out