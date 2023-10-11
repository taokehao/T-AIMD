#!/usr/bin/env python3
# encoding: utf-8
"""
@Time    : 2023/4/11
@Author  : Kehao Tao
@File    : transformer.py
@Software: PyCharm
@desc: transformer架构
"""
import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return self.dropout(x)


class TransformerTS(nn.Module):
    def __init__(self,
                 initial_dim,
                 middle_dim,
                 d_model,
                 nhead,
                 num_encoder_layers,
                 num_decoder_layers,
                 dim_feedforward,
                 dropout):

        super(TransformerTS, self).__init__()
        '''位置编码层'''
        self.pos = PositionalEncoding(d_model)
        '''embedding层'''
        # self.enc_input_fc = nn.Linear(initial_dim, middle_dim)
        # self.dec_input_fc = nn.Linear(initial_dim, middle_dim)
        # self.enc_input_fc = nn.Embedding(1, d_model)
        self.embedding = nn.Linear(1, d_model)
        '''transformer的模型搭建'''
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                    dropout=dropout)  # 实例化官方"编码器层"
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_encoder_layers)  # 实例化官方"编码器"
        decoder_layers = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                    dropout=dropout)  # 实例化官方"解码器层"
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layers, num_layers=num_decoder_layers)  # 实例化官方"解码器"
        '''最后的线性层'''
        self.out_fc = nn.Linear(initial_dim*d_model, initial_dim)

    def forward(self, x):

        x = x.transpose(0, 2)
        '''embedding'''
        # embed_encoder_input = self.enc_input_fc(x).transpose(0, 2)
        embed_encoder_input = self.pos(self.embedding(x))
        # embed_decoder_input = self.dec_input_fc(x).transpose(0, 2)
        embed_decoder_input = self.pos(self.embedding(x))
        '''transform'''
        x = self.encoder(src=embed_encoder_input)
        feature_encoder = x.contiguous()
        x = self.decoder(tgt=embed_decoder_input, memory=x)

        '''output'''
        x = x.transpose(0, 1).flatten(start_dim=1, end_dim=2)
        # x = self.out_fc(x.flatten(start_dim=1))
        x = self.out_fc(x)

        return x, feature_encoder.transpose(0, 1).flatten(start_dim=1, end_dim=2)
