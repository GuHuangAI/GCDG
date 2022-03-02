# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


""" Define the Seq2Seq Generation Network """
import math
import numpy as np
import torch
import torch.nn as nn
import models.smpl as smpl_model
from utils.rotation import rotation_matrix_to_angle_axis
from models.layers import MultiHeadAttention, PositionwiseFeedForward
# from configs.configs_cmtr import config
# smpl_model_path = config.smpl_model_path
# train_batch_size = config.train_batch_size

def get_non_pad_mask(seq):
    assert seq.dim() == 3
    non_pad_mask = torch.abs(seq).sum(2).ne(0).type(torch.float)
    return non_pad_mask.unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """
    len_q = seq_q.size(1)
    padding_mask = torch.abs(seq_k).sum(2).eq(0)  # sum the vector of last dim and then judge
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_subsequent_mask(seq, sliding_windown_size):
    """ For masking out the subsequent info. """
    batch_size, seq_len, _ = seq.size()
    mask = torch.ones((seq_len, seq_len), device=seq.device, dtype=torch.uint8)
    mask = torch.triu(mask, diagonal=-sliding_windown_size)
    mask = torch.tril(mask, diagonal=sliding_windown_size)
    mask = 1 - mask
    # print(mask)
    return mask.bool()


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None, non_pad_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        # enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        # enc_output *= non_pad_mask

        return enc_output, enc_slf_attn

class DecoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_input, slf_attn_mask=None, non_pad_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            dec_input, enc_input, enc_input, mask=slf_attn_mask)
        # enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        # enc_output *= non_pad_mask

        return enc_output, enc_slf_attn

class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self, max_seq_len=1800, input_size=20, d_word_vec=10,
            n_layers=6, n_head=8, d_k=64, d_v=64,
            d_model=800, d_inner=256, dropout=0.1):

        super().__init__()

        self.d_model = d_model
        n_position = max_seq_len + 1

        self.src_emb = nn.Linear(input_size, d_word_vec)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, mask=None, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.src_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=mask)

            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class CMTr(nn.Module): ##Cross Modal Transformer
    def __init__(self, d_in=800, d_hidden=800, d_out=219, d_forward=1024, dropout=0.1, num_queries=120,
                 n_layers=6, n_head=8, d_k=64, d_v=64, max_seq_len=1800):
        super().__init__()
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.layers = nn.ModuleList([
            DecoderLayer(d_hidden, d_forward, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.enc_emb = nn.Linear(d_in, d_hidden)
        n_position = max_seq_len + 1
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_hidden, padding_idx=0),
            freeze=True)
        self.tgt_emb = nn.Linear(d_out, d_hidden)
        # self.query_embed = nn.Embedding(num_queries, d_hidden)
        self.pos_q = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, num_queries+1, d_hidden)))
        # self.query_embed = nn.Embedding(num_queries, d_hidden)
        # nn.init.normal_(self.query_embed.weight)
        # self.dropout = nn.Dropout(dropout)
        # self.out_layer = nn.Linear(d_hidden, d_out)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                nn.init.xavier_uniform_(m.weight, 1.0)

    def forward(self, enc_input, enc_outputs_pos, tgt, mask=None, return_attns=False, learnable_query=False):
        enc_slf_attn_list = []
        bs = enc_input.shape[0]
        src = self.enc_emb(enc_input) + self.position_enc(enc_outputs_pos)
        if learnable_query:
            tgt = self.tgt_emb(tgt)
        else:
            tgt = self.tgt_emb(tgt) + self.pos_q.repeat(bs, 1, 1)
        bs, _, _ = src.shape
        # if tgt == None:
        #     tgt = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1) + self.pos.unsqueeze(0).repeat(bs, 1, 1)
        # else:
        #     tgt = tgt + self.pos.unsqueeze(0).repeat(bs, 1, 1)
        for dec_layer in self.layers:
            tgt, enc_slf_attn = dec_layer(tgt, src, slf_attn_mask=mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        # output = self.out_layer(self.dropout(tgt))
        output = tgt
        if return_attns:
            return output, enc_slf_attn_list
        return output

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

        # 鍘熷垵濮?
        # BOP = BOS_POSE
        # BOP = np.tile(BOP, (bsz, 1))
        # root = BOP[:, 2*8:2*9]
        # bos = BOP - np.tile(root, (1, 25))
        # bos[:, 2*8:2*9] = root
        # out_frame = torch.from_numpy(bos).float().to(device)
        # out_seq = torch.FloatTensor(bsz, 1).to(device)

        return (vec_h, vec_c)  # , out_frame, out_seq
        # return (vec_h, vec_c), out_frame, out_seq

    def forward(self, in_frame, vec_h, vec_c):
        in_frame = self.tgt_emb(in_frame)
        in_frame = self.dropout(in_frame)

        vec_h0, vec_c0 = self.lstm1(in_frame, (vec_h[0], vec_c[0]))
        # vec_h1, vec_c1 = self.lstm2(vec_h[0], (vec_h[1], vec_c[1]))
        # vec_h2, vec_c2 = self.lstm3(vec_h[1], (vec_h[2], vec_c[2]))
        vec_h1, vec_c1 = self.lstm2(vec_h0, (vec_h[1], vec_c[1]))
        vec_h2, vec_c2 = self.lstm3(vec_h1, (vec_h[2], vec_c[2]))

        vec_h_new = [vec_h0, vec_h1, vec_h2]
        vec_c_new = [vec_c0, vec_c1, vec_c2]
        return vec_h2, vec_h_new, vec_c_new


class Model(nn.Module):
    def __init__(self, encoder1, encoder2, decoder, condition_step=10,
                 d_hidden=800, num_queries=120, d_out=219,
                 sliding_windown_size=100, lambda_v=0.01, device=None, config=None):
        super().__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.decoder = decoder
        # self.in_layer = nn.Linear(d_out, d_hidden)
        self.query_embed = nn.Embedding(num_queries + 1, d_out)
        # self.
        # self.pos = nn.Parameter(torch.randn(20, d_out))
        nn.init.normal_(self.query_embed.weight, std=0.05)
        self.dropout = nn.Dropout(0.1)
        self.out_layer = nn.Linear(d_hidden, d_out)
        self.genre_token = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, 1, d_out)))
        self.genre_layer = nn.Sequential(
            nn.Linear(d_hidden, 512),
            nn.Dropout(0.1),
            nn.Linear(512, 10)
            )
        # self.linear = nn.Linear(decoder.hidden_size + encoder1.d_model, decoder.input_size)

        self.condition_step = condition_step
        self.sliding_windown_size = sliding_windown_size
        self.lambda_v = lambda_v
        self.device = device

        self.smpl_model = smpl_model.create(config.smpl_model_path, \
                                            batch_size=config.train_batch_size, model_type='smpl', gender='neutral',
                                            use_face_contour=False, ext='npz', flat_hand_mean=True, use_pca=False)

    def init_decoder_hidden(self, bsz):
        return self.decoder.init_state(bsz, self.device)

    # dynamic auto-condition + self-attention mask
    def forward(self, audio_seq, condition_motion, tgt_seq, regressive_len=20, type='auto', eval=False):
        if type == 'full':
            out_seq = self.full_forward(audio_seq,condition_motion,tgt_seq, eval=eval)
        elif type == 'auto':
            assert regressive_len is not None
            out_seq = self.auto_forward(audio_seq, condition_motion, tgt_seq, regressive_len=regressive_len, eval=eval)
        else:
            raise NotImplementedError
        return out_seq

    def auto_forward(self, audio_seq, condition_motion, tgt_seq, regressive_len=20, eval=False): ### autoregressive
        # print(src_seq[0].shape)
        bsz, out_len, _ = tgt_seq.size()
        loop_time = math.ceil(out_len//regressive_len)
        # vec_h, vec_c = hidden

        '''audio & embedding'''
        src_seq = audio_seq[0]
        src_pos1 = audio_seq[1]
        enc_mask1 = get_subsequent_mask(src_seq, self.sliding_windown_size)
        enc_outputs1, *_ = self.encoder1(src_seq, src_pos1, mask=enc_mask1)  # (b, seq_len1, dim)

        '''motion & embedding'''
        condition_seq = condition_motion[0]
        src_pos2 = condition_motion[1]
        enc_mask2 = get_subsequent_mask(condition_seq, self.sliding_windown_size)
        enc_outputs2, *_ = self.encoder2(condition_seq, src_pos2, mask=enc_mask2)  # (b, seq_len2, dim)

        enc_outputs = torch.cat([enc_outputs1, enc_outputs2], 1)  # (b, seq_len1+seq_len2, dim)
        enc_outputs_pos = np.array([[pos_i + 1 for pos_i, v_i in enumerate(inst)] for inst in enc_outputs])
        enc_outputs_pos = torch.LongTensor(enc_outputs_pos).to(enc_outputs.device)
        out_seqs = []
        # out_query = self.query_embed.weight.unsqueeze(0).repeat(bsz, 1, 1) + self.pos.unsqueeze(0).repeat(bsz, 1, 1)
        # out_query = out_query[:, :regressive_len, :]
        # out_query = self.in_layer(condition_motion[:, -regressive_len:, :]) + self.pos.unsqueeze(0).repeat(bsz, 1, 1)
        out_query = condition_seq[:, -regressive_len:, :]
        for l in range(loop_time):
            out_seq = self.decoder(enc_outputs, enc_outputs_pos, out_query)
            out_seq = self.out_layer(self.dropout(out_seq))
            out_query = out_seq.detach()
            out_seqs.append(out_seq)
        out_seq = torch.cat(out_seqs, dim=1)
        # out_seq = self.out_layer(self.dropout(out_seqs))
        if eval:
            return out_seq
        out_seq = self.post_process(out_seq, targets=tgt_seq)
        return out_seq

    def full_forward(self, audio_seq, condition_motion, tgt_seq, eval=False):  ### non-autoregressive
        # print(src_seq[0].shape)
        bsz, seq_len, _ = tgt_seq.size()
        # vec_h, vec_c = hidden

        '''audio & embedding'''
        src_seq = audio_seq[0]
        src_pos1 = audio_seq[1]
        enc_mask1 = get_subsequent_mask(src_seq, self.sliding_windown_size)
        enc_outputs1, *_ = self.encoder1(src_seq, src_pos1, mask=None)  # (b, seq_len1, dim)

        '''motion & embedding'''
        condition_seq = condition_motion[0]
        src_pos2 = condition_motion[1]
        # enc_mask2 = get_subsequent_mask(condition_seq, self.sliding_windown_size)
        # enc_outputs2, *_ = self.encoder2(condition_seq, src_pos2, mask=enc_mask2)  # (b, seq_len2, dim)
        #
        # enc_outputs = torch.cat([enc_outputs1, enc_outputs2], 1)  # (b, seq_len1+seq_len2, dim)
        # enc_outputs_pos = np.array([[pos_i + 1 for pos_i, v_i in enumerate(inst)] for inst in enc_outputs])
        # enc_outputs_pos = torch.LongTensor(enc_outputs_pos).to(enc_outputs.device)
        # out_query = self.query_embed.weight.unsqueeze(0).repeat(bsz, 1, 1)
        out_query = torch.cat([condition_seq[:, :120, :], self.genre_token.repeat(bsz, 1, 1)], dim=1)
        out_seq = self.decoder(enc_outputs1, src_pos1, out_query, learnable_query=False)
        out_tmp = self.out_layer(self.dropout(out_seq[:, :120, :]))
        # out_genre = out_tmp[:, -1, :]
        genre = self.genre_layer(out_seq[:, -1, :])
        out_seq = out_tmp
        if eval:
            return out_seq
        out_seq = self.post_process(out_seq, targets=tgt_seq)
        out_seq.append(genre)
        return out_seq

    def generate(self, audio_seq, condition_motion, tgt_seq, regressive_len=10, target_len=20*60, type='full'):
        print('regressive_len is ', regressive_len)
        bsz, out_len, _ = tgt_seq.size()
        loop_time = math.ceil(target_len // regressive_len)
        # vec_h, vec_c = hidden
        for i in range(loop_time):
            '''audio & embedding'''
            src_seq = audio_seq[:, i*regressive_len:i*regressive_len+240, :]
            src_pos1 = np.array([[pos_i + 1 for pos_i, v_i in enumerate(inst)] for inst in src_seq])
            src_pos1 = torch.LongTensor(src_pos1).to(src_seq.device)
            enc_mask1 = get_subsequent_mask(src_seq, self.sliding_windown_size)
            enc_outputs1, *_ = self.encoder1(src_seq, src_pos1, mask=None)  # (b, seq_len1, dim)

            '''motion & embedding'''
            condition_seq = condition_motion[:, i*regressive_len:i*regressive_len+120, :]
            src_pos2 = np.array([[pos_i + 1 for pos_i, v_i in enumerate(inst)] for inst in condition_seq])
            # src_pos2 = torch.LongTensor(src_pos2).to(condition_seq.device)
            # enc_mask2 = get_subsequent_mask(condition_seq, self.sliding_windown_size)
            # enc_outputs2, *_ = self.encoder2(condition_seq, src_pos2, mask=enc_mask2)  # (b, seq_len2, dim)
            #
            # enc_outputs = torch.cat([enc_outputs1, enc_outputs2], 1)  # (b, seq_len1+seq_len2, dim)
            # enc_outputs_pos = np.array([[pos_i + 1 for pos_i, v_i in enumerate(inst)] for inst in enc_outputs])
            # enc_outputs_pos = torch.LongTensor(enc_outputs_pos).to(enc_outputs.device)
            if type == 'auto':
                out_seqs = []
                # out_query = self.query_embed.weight.unsqueeze(0).repeat(bsz, 1, 1) + self.pos.unsqueeze(0).repeat(bsz, 1, 1)
                # out_query = out_query[:, :regressive_len, :]
                # out_query = self.in_layer(condition_motion[:, -regressive_len:, :]) + self.pos.unsqueeze(0).repeat(bsz, 1, 1)
                out_query = condition_seq[:, -regressive_len:, :]
                for l in range(120//regressive_len):
                    out_seq = self.decoder(enc_outputs, enc_outputs_pos, out_query)
                    out_seq = self.out_layer(self.dropout(out_seq))
                    out_query = out_seq.detach()
                    out_seqs.append(out_seq)
                out_seq = torch.cat(out_seqs, dim=1)
            elif type == 'full':
                out_query = torch.cat([condition_seq[:, :120, :], self.genre_token.repeat(bsz, 1, 1)], dim=1)
                out_seq = self.decoder(enc_outputs1, src_pos1, out_query, learnable_query=False)
                out_tmp = self.out_layer(self.dropout(out_seq[:, :120, :]))
                genre = self.genre_layer(out_seq[:, -1, :])
                # print(genre)
                out_seq = out_tmp
                # out_seq = out_tmp + condition_seq[:, -1, :].unsqueeze(1).repeat(1, 120, 1)
            condition_motion = torch.cat([condition_motion, out_seq[:, :regressive_len, :]], dim=1)

        return condition_motion[:, 120:, :]

    def smpl_forward(self, pose):
        rot9d = pose.reshape(-1, 24, 9).reshape(-1, 24, 3, 3).reshape(-1, 3, 3)
        pred_pose = rotation_matrix_to_angle_axis(rot9d).reshape(-1, 24 * 3)
        params_dict = {'global_orient': pred_pose[:, :3], 'body_pose': pred_pose[:, 3:],
                       'betas': torch.zeros(pred_pose.shape[0], 10, device=rot9d.device).float()}
        smpl_out = self.smpl_model(**params_dict, return_verts=True, return_full_pose=True)

        return smpl_out

    def post_process(self, output, targets=None):
        output_pose = output[:, :, :216]
        pred_smpl_out = self.smpl_forward(output_pose)
        pred_joints = pred_smpl_out.joints[:, :24]
        out = [output, pred_joints]

        if targets is not None:
            targets = targets[:, :, :216]
            target_smpl_out = self.smpl_forward(targets)
            target_joints = target_smpl_out.joints[:, :24]
            out.append(target_joints)

        return out

if __name__ == '__main__':
    pos = get_sinusoid_encoding_table(100, 200, padding_idx=0)
    pass