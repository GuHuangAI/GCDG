# -*- coding: utf-8 -*-
import os
import sys
import re
import datetime
import numpy


def get_network(args, device=None):
    if args.network == "2tcnlstm":
        from models.TCN2_LSTM1 import TemporalConvNet, Decoder, Model
        encoder1 = TemporalConvNet(num_inputs=200,
                                   num_channels=[200, 200, 200, 200],
                                   input_size=438,
                                   d_word_vec=200, )

        encoder2 = TemporalConvNet(num_inputs=200,
                                   num_channels=[200, 200, 200, 200],
                                   input_size=219,
                                   d_word_vec=200, )

        decoder = Decoder(input_size=219,
                          d_word_vec=219,
                          hidden_size=1024,
                          encoder_d_model=200,
                          dropout=0.1)

        net = Model(encoder1, encoder2, decoder,
                    condition_step=10,
                    sliding_windown_size=100,
                    lambda_v=0.02,
                    device=device)

    elif args.network == 'Dancerevolution_2Transformer':
        from models.dancerevolution_2Transformer import Encoder, Decoder, Model
        encoder1 = Encoder(max_seq_len=4500,
                           input_size=438,
                           d_word_vec=200,
                           n_layers=2,
                           n_head=8,
                           d_k=64,
                           d_v=64,
                           d_model=200,
                           d_inner=1024,
                           dropout=0.1)

        encoder2 = Encoder(max_seq_len=4500,
                           input_size=219,
                           d_word_vec=200,
                           n_layers=2,
                           n_head=8,
                           d_k=64,
                           d_v=64,
                           d_model=200,
                           d_inner=1024,
                           dropout=0.1)

        decoder = Decoder(input_size=219,
                          d_word_vec=219,
                          hidden_size=1024,
                          encoder_d_model=200,
                          dropout=0.1)

        net = Model(encoder1, encoder2, decoder,
                    condition_step=10,
                    sliding_windown_size=100,
                    lambda_v=0.02,
                    device=device)
    elif args.network == 'CrossModalTr':
        model_py = 'models.cross_modal_tr_v' + str(args.saved_version)
        print(args.saved_version)
        pa = __import__(model_py, fromlist=('Encoder', 'CMTr', 'Model'))
        Encoder, CMTr, Model = pa.Encoder, pa.CMTr, pa.Model
        # from models.cross_modal_tr_v6 import Encoder, CMTr, Model
        if args.saved_version in ['6', '8', '9', '10']:
            encoder1 = Encoder(max_seq_len=4500,
                               input_size=438,
                               d_word_vec=256,
                               n_layers=6,
                               n_head=8,
                               d_k=64,
                               d_v=64,
                               d_model=256,
                               d_inner=1024,
                               dropout=0.1)

            encoder2 = Encoder(max_seq_len=4500,
                               input_size=219,
                               d_word_vec=256,
                               n_layers=2,
                               n_head=8,
                               d_k=64,
                               d_v=64,
                               d_model=256,
                               d_inner=1024,
                               dropout=0.1)
            decoder = CMTr(d_in=256,
                           d_hidden=512,
                           d_out=219,
                           d_forward=1024,
                           dropout=0.1,
                           num_queries=120,
                           n_layers=6,
                           n_head=8,
                           d_k=64,
                           d_v=64,
                           max_seq_len=4500, )
        elif args.saved_version in ['4', '5', '7', ]:
            encoder1 = Encoder(max_seq_len=4500,
                               input_size=35,
                               d_word_vec=256,
                               n_layers=2,
                               n_head=8,
                               d_k=64,
                               d_v=64,
                               d_model=256,
                               d_inner=1024,
                               dropout=0.1)

            encoder2 = Encoder(max_seq_len=4500,
                               input_size=219,
                               d_word_vec=256,
                               n_layers=2,
                               n_head=8,
                               d_k=64,
                               d_v=64,
                               d_model=256,
                               d_inner=1024,
                               dropout=0.1)
            decoder = CMTr(d_in=256,
                           d_hidden=512,
                           d_out=219,
                           d_forward=1024,
                           dropout=0.1,
                           num_queries=120,
                           n_layers=6,
                           n_head=8,
                           d_k=64,
                           d_v=64,
                           max_seq_len=4500, )
        elif args.saved_version == '6_2':
            print('111')
            encoder1 = Encoder(max_seq_len=4500,
                               input_size=35,
                               d_word_vec=256,
                               n_layers=2,
                               n_head=8,
                               d_k=64,
                               d_v=64,
                               d_model=256,
                               d_inner=1024,
                               dropout=0.1)

            encoder2 = Encoder(max_seq_len=4500,
                               input_size=219,
                               d_word_vec=256,
                               n_layers=2,
                               n_head=8,
                               d_k=64,
                               d_v=64,
                               d_model=256,
                               d_inner=1024,
                               dropout=0.1)
            decoder = CMTr(d_in=256,
                           d_hidden=512,
                           d_out=219,
                           d_forward=1024,
                           dropout=0.1,
                           num_queries=120,
                           n_layers=10,
                           n_head=8,
                           d_k=64,
                           d_v=64,
                           max_seq_len=4500, )
        elif args.saved_version == '12' or args.saved_version == '14':
            encoder1 = Encoder(max_seq_len=4500,
                               input_size=35,
                               d_word_vec=256,
                               n_layers=2,
                               n_head=8,
                               d_k=64,
                               d_v=64,
                               d_model=256,
                               d_inner=1024,
                               dropout=0.1)

            encoder2 = Encoder(max_seq_len=4500,
                               input_size=219,
                               d_word_vec=256,
                               n_layers=2,
                               n_head=8,
                               d_k=64,
                               d_v=64,
                               d_model=256,
                               d_inner=1024,
                               dropout=0.1)
            decoder = CMTr(d_in=256,
                           d_hidden=512,
                           d_out=219,
                           d_forward=1024,
                           dropout=0.1,
                           num_queries=60,
                           n_layers=10,
                           n_head=8,
                           d_k=64,
                           d_v=64,
                           max_seq_len=4500, )
        elif args.saved_version == '0':
            encoder1 = Encoder(max_seq_len=4500,
                               input_size=35,
                               d_word_vec=800,
                               n_layers=2,
                               n_head=10,
                               d_k=80,
                               d_v=80,
                               d_model=800,
                               d_inner=800,
                               dropout=0.1,
                               seq_len=240)

            encoder2 = Encoder(max_seq_len=4500,
                               input_size=219,
                               d_word_vec=800,
                               n_layers=2,
                               n_head=10,
                               d_k=80,
                               d_v=80,
                               d_model=800,
                               d_inner=800,
                               dropout=0.1,
                               seq_len=120)
            decoder = Encoder(max_seq_len=4500,
                              input_size=800,
                              d_word_vec=800,
                              n_layers=12,
                              n_head=10,
                              d_k=80,
                              d_v=80,
                              d_model=800,
                              d_inner=800,
                              dropout=0.1)

        net = Model(encoder1, encoder2, decoder,
                    condition_step=10,
                    # num_queries=60,
                    sliding_windown_size=100,
                    d_hidden=512,
                    lambda_v=0.02,
                    device=device,
                    config=args.config)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    return net
