# coding = utf-8
"""
作者   : Hilbert
时间   :2022/8/16 9:23
"""
import sys
import os
from warnings import simplefilter

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]  # 上一级目录
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)
simplefilter(action='ignore', category=Warning)
simplefilter(action='ignore', category=FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import weight_norm

class Informer(nn.Module):
    def __init__(self, enc_in, c_out,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 lstm_hidden_size=256, lstm_hidden_layers=3, reg_size=32,
                 device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=lstm_hidden_size, num_layers=lstm_hidden_layers,
                            batch_first=True)

        self.regression = nn.Sequential(nn.Linear(in_features=lstm_hidden_size, out_features=reg_size),
                                        nn.ReLU(),
                                        nn.Linear(in_features=reg_size, out_features=c_out))


    def forward(self, x_enc):
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # print(enc_out.shape)

        dec_out, _ = self.lstm(enc_out)
        # print(dec_out.shape)
        dec_out = dec_out[:, -1, :]
        out = self.regression(dec_out)

        if self.output_attention:
            return out.view(-1), attns
        else:
            return out.view(-1)


class Informer_CNN(nn.Module):
    def __init__(self, enc_in, c_out, window_size,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True, cnn_kernel_size=3, reg_size=32,
                 device=torch.device('cuda:0')):
        super(Informer_CNN, self).__init__()
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        cnn_channels = int(window_size/(2**d_layers))
        self.cnn1d = nn.Sequential(nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels,
                                             kernel_size=cnn_kernel_size, padding='same'),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=4, stride=4),
                                   nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels,
                                             kernel_size=cnn_kernel_size, padding='same'),
                                   # nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=4, stride=4),
                                   # nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=3,
                                   #           padding='same'),
                                   # nn.MaxPool1d(kernel_size=4, stride=4),
                                   )

        self.regression = nn.Sequential(nn.Linear(in_features=int(cnn_channels*d_model/16), out_features=reg_size, bias=False),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(in_features=reg_size, out_features=c_out))

    def forward(self, x_enc):
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # print(enc_out.shape)

        dec_out = self.cnn1d(enc_out)
        # print(dec_out.shape)
        dec_out = dec_out.view(dec_out.shape[0], -1)
        out = self.regression(dec_out)

        if self.output_attention:
            return out.view(-1), attns
        else:
            return out.view(-1)


class SourceTransferInformer_CNN(nn.Module):
    def __init__(self, enc_in, c_out, window_size, transfer_fcn_size,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True, cnn_kernel_size=3, reg_size=32,
                 device=torch.device('cuda:0')):
        super(SourceTransferInformer_CNN, self).__init__()
        self.attn = attn
        self.output_attention = output_attention

        # source transfer with fully connected layers
        # self.source_transfer = nn.Sequential(nn.Linear(in_features=window_size, out_features=transfer_fcn_size),
        #                                      nn.ReLU(),
        #                                      nn.Linear(in_features=transfer_fcn_size, out_features=window_size),
        #                                      )
        self.source_transfer = nn.LSTM(input_size=enc_in, hidden_size=enc_in*4,
                                     num_layers=2,
                                     batch_first=True)
        self.fcn = nn.Linear(in_features=enc_in*4, out_features=enc_in)

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        cnn_channels = int(window_size/(2**d_layers))
        self.cnn1d = nn.Sequential(nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels,
                                             kernel_size=cnn_kernel_size, padding='same'),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=4, stride=4),
                                   nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels,
                                             kernel_size=cnn_kernel_size, padding='same'),
                                   # nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=4, stride=4),
                                   # nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=3,
                                   #           padding='same'),
                                   # nn.MaxPool1d(kernel_size=4, stride=4),
                                   )

        self.regression = nn.Sequential(nn.Linear(in_features=int(cnn_channels*d_model/16), out_features=reg_size, bias=False),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(in_features=reg_size, out_features=c_out))

    def forward(self, x_enc):
        # x_enc = x_enc.permute(0, 2, 1)
        # # new_x_enc = self.source_transfer(x_enc)
        # # x_enc = x_enc + new_x_enc
        # # x_enc = x_enc.permute(0, 2, 1)
        # new_x_enc = self.source_transfer(x_enc)
        # # new_x_enc = new_x_enc + x_enc[:, :, :2]
        # # voltage_diff = new_x_enc[:, :, 0] - new_x_enc[:, 0, 0].reshape(-1, 1)
        # new_x_enc = new_x_enc + x_enc
        # voltage_diff = new_x_enc[:, 0, :] - new_x_enc[:, 0, 0].reshape(-1, 1)
        # divided_voltage = voltage_diff.clone()
        # divided_voltage[divided_voltage < 1e-7] = 1e10
        # delta_Q_V = new_x_enc[:, 1, :] / divided_voltage
        # x_enc = torch.cat([new_x_enc[:, :2, :], voltage_diff.reshape(voltage_diff.shape[0], -1, voltage_diff.shape[1]),
        #                   delta_Q_V.reshape(delta_Q_V.shape[0], -1, delta_Q_V.shape[1])], axis=1)
        # x_enc = x_enc.permute(0, 2, 1)
        new_x_enc, _ = self.source_transfer(x_enc)
        new_x_enc = self.fcn(new_x_enc)
        x_enc += new_x_enc

        # new_x_enc = self.source_transfer(x_enc)
        # x_enc = x_enc + new_x_enc
        # b = x_enc[0].detach().numpy()
        # c = new_x_enc[0].detach().numpy()
        # d = x_enc[0].detach().numpy()
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # print(enc_out.shape)

        dec_out = self.cnn1d(enc_out)
        # print(dec_out.shape)
        dec_out = dec_out.view(dec_out.shape[0], -1)
        out = self.regression(dec_out)

        if self.output_attention:
            return out.view(-1), attns
        else:
            return out.view(-1)

    def transfer_input(self, x_enc):
        new_x_enc, _ = self.source_transfer(x_enc)
        new_x_enc = self.fcn(new_x_enc)
        x_enc += new_x_enc
        # x_enc = x_enc.permute(0, 2, 1)
        # x_enc = self.source_transfer(x_enc)
        # x_enc = x_enc.permute(0, 2, 1)
        return x_enc

class SNL_Informer_CNN(nn.Module):
    def __init__(self, enc_in, c_out, window_size,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True, cnn_kernel_size=3, reg_size=32,
                 device=torch.device('cuda:0')):
        super(SNL_Informer_CNN, self).__init__()
        self.attn = attn
        self.output_attention = output_attention

        self.transformation = nn.Linear(in_features=window_size, out_features=window_size)
        self.relu = nn.ReLU()
        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        cnn_channels = int(window_size/(2**d_layers))
        self.cnn1d = nn.Sequential(nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels,
                                             kernel_size=cnn_kernel_size, padding='same'),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=4, stride=4),
                                   nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels,
                                             kernel_size=cnn_kernel_size, padding='same'),
                                   # nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=4, stride=4),
                                   # nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=3,
                                   #           padding='same'),
                                   # nn.MaxPool1d(kernel_size=4, stride=4),
                                   )

        self.regression = nn.Sequential(nn.Linear(in_features=int(cnn_channels*d_model/16), out_features=reg_size, bias=False),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(in_features=reg_size, out_features=c_out))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x_enc = self.relu(self.transformation(x))
        x_enc = x_enc.permute(0, 2, 1)
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # print(enc_out.shape)

        dec_out = self.cnn1d(enc_out)
        # print(dec_out.shape)
        dec_out = dec_out.view(dec_out.shape[0], -1)
        out = self.regression(dec_out)

        if self.output_attention:
            return out.view(-1), attns
        else:
            return out.view(-1)


class Informer_TCN(nn.Module):
    def __init__(self, enc_in, c_out, window_size,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.1, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True, num_levels=3, reg_size=32,
                 device=torch.device('cuda:0')):
        super(Informer_TCN, self).__init__()
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        tcn_channels = int(window_size/(2**d_layers))
        layers = []
        num_channels = [tcn_channels, int(tcn_channels/2), int(tcn_channels/4), int(tcn_channels/8)]
        kernel_size = 2
        for i in range(num_levels):
            dilation_size = 2 ** i      # dilation_size 根据层级数指数增加
            in_channels = tcn_channels if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            # 从num_channels中抽取每一个残差模块的输入通道数与输出通道数
            # 调用残差模块
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        # 将所有残差模块堆叠起来组成一个深度卷积网络
        self.tcn_network = nn.Sequential(*layers)


        self.regression = nn.Sequential(nn.Linear(in_features=int(int(tcn_channels/4)*d_model), out_features=reg_size),
                                        nn.ReLU(),
                                        nn.Linear(in_features=reg_size, out_features=c_out))

    def forward(self, x_enc):
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # print(enc_out.shape)

        dec_out = self.tcn_network(enc_out)
        # print(dec_out.shape)
        dec_out = dec_out.view(dec_out.shape[0], -1)
        out = self.regression(dec_out)

        if self.output_attention:
            return out.view(-1), attns
        else:
            return out.view(-1)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)

        return self.dropout(x)

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        # transformer中的位置编码
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 取出对应的位置编码
        return self.pe[:, :x.size(1)]


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        # if self.mask_flag:
        #     if attn_mask is None:
        #         attn_mask = TriangularCausalMask(B, L, device=queries.device)
        #
        #     scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _prob_QK_v1(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        Q_K_sample = torch.matmul(Q, K.transpose(-2, -1))
        # K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        # K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        # Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        # if self.mask_flag:
        #     attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
        #     scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK_v1(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm2(x+y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)
        return x


# TCN
# reference: https://github.com/hilbert-qyw/TCN/blob/master/TCN/tcn.py
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        将后边多扩展的大小为padding的长度给删除，通过增加padding方式对卷积后的张量做切片而实现因果卷积（只能补在一边，不能补在另一边)
        tensor.contiguous() 会返回有连续内存的相同张量， 有些tensor并不是占用一整块内存，而是由不同的数据块组成
        而tensor的view()操作依赖于内存是整块，这时候就需要执行contiguous()这个函数，把tensor变成在内存中连续分布的形式
        """
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size


    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """

        super(TemporalBlock, self).__init__()

        # 定义第一个扩散卷积层，扩散是dilation
        # 重参数将权重向量w用了两个独立的参数表示其幅度和方向，加速收敛
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))

        # 根据第一个卷积层的输出与padding大小实现因果卷积
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        # 在先前的输出结果上添加激活函数与dropout完成第一个卷积

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        # padding保证了输入序列与输出序列的长度相同，且padding只对序列的一边进行操作
        # 卷积前后的通道不一定一致
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 将卷积模块所有部件拼接起来
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # point-wise 1*1 卷积，实现通道融合，将通道变换至输入相同，实现残差连接
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # 输入x进行空洞卷积、因果卷积
        out = self.net(x)

        # 输入x进行通道融合，使两个路径输出的数据维度相同，从而进行残差连接
        res = x if self.downsample is None else self.downsample(x)

        # 残差连接并返回
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self,
                 num_inputs,
                 num_channels,
                 kernel_size=2,
                 dropout=0,
                 linear_hidden_1=256,
                 linear_hidden_2=64,
                 predict_sensor=9):
        """
        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """

        super(TCN, self).__init__()
        layers = []
        # num_channels 为各层卷积运算的输出通道数或卷积核数量
        # num_channels 的长度即需要执行的卷积层数量
        num_levels = len(num_channels)
        # 扩张系数随网络层级的增加而成指数级增加，增大感受野并不丢弃任何输入序列的元素
        for i in range(num_levels):
            dilation_size = 2 ** i      # dilation_size 根据层级数指数增加
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            # 从num_channels中抽取每一个残差模块的输入通道数与输出通道数
            # 调用残差模块
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        # 将所有残差模块堆叠起来组成一个深度卷积网络
        self.network = nn.Sequential(*layers)
        # shape: [bs, num_channels, seq_len]
        self.regression = nn.Sequential(nn.Linear(in_features=num_channels[-1], out_features=linear_hidden_1),
                                        nn.ReLU(),
                                        nn.Linear(in_features=linear_hidden_1, out_features=linear_hidden_2,),
                                        nn.ReLU(),
                                        nn.Linear(in_features=linear_hidden_2, out_features=predict_sensor),
                                        )

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。
        x: [bs, channels, seq_len]
        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        x = x.permute(0, 2, 1)
        tcn_out = self.network(x)
        out = self.regression(tcn_out[:, :, -1])

        return out

if __name__ == '__main__':
    input = torch.randn((1000, 200, 4))
    target = torch.randn(1000)
    print(input.shape)

    sample_dataset = TensorDataset(input, target)
    dataloader = DataLoader(sample_dataset, batch_size=32, shuffle=True)

    # model
    model = Informer(enc_in=4, c_out=1,
                     factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                     dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                     output_attention=False, distil=True, mix=True,
                     device=torch.device('cuda:0'))


    for sam, tar in dataloader:
        pre = model(sam)
        print(pre)

