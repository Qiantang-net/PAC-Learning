import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from utils.utils import norm, cut_add_paste, adjust_amplitude_in_frequency_domain, add_gaussian_noise, add_random_mask, \
    add_gamma_noise, add_rayleigh_noise, add_exponential_noise, add_uniform_noise
from .attn_layer import AttentionLayer
from .embedding import TokenEmbedding, InputEmbedding
from .loss_functions import temporal_contrastive_loss, instance_contrastive_loss

# ours
from .ours_memory_module import MemoryModule, Memory_Unit, AMemoryModule  # ,Memory_Unit

import torch
import torch.nn.functional as F



class ADCLS_head(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim,128), nn.ReLU(), nn.Linear(128,out_dim), nn.Sigmoid())
    def forward(self, x):
        return self.mlp(x)

class EncoderLayer(nn.Module):
    def __init__(self, attn, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.attn_layer = attn
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x):
        '''
        x : N x L x C(=d_model)
        '''
        out = self.attn_layer(x)
        x = x + self.dropout(out)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)    # N x L x C(=d_model)
    

# Transformer Encoder
class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        '''
        x : N x L x C(=d_model)
        '''
        for attn_layer in self.attn_layers:
            x = attn_layer(x)

        if self.norm is not None:
            x = self.norm(x)

        return x


class Encoder_a(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder_a, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        '''
        x : N x L x C(=d_model)
        '''
        for attn_layer in self.attn_layers:
            x = attn_layer(x)

        if self.norm is not None:
            x = self.norm(x)

        return x


    
class Decoder(nn.Module):
    def __init__(self, d_model, c_out, d_ff=None, activation='relu', dropout=0.1):
        super(Decoder, self).__init__()
        # self.decoder_layer = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=2,
        #                              batch_first=True, bidirectional=True)
        self.out_linear = nn.Linear(d_model, c_out)
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.decoder_layer1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)

        # self.decoder_layer_add = nn.Conv1d(in_channels=d_ff, out_channels=d_ff, kernel_size=1)

        self.decoder_layer2 = nn.Conv1d(in_channels=d_ff, out_channels=c_out, kernel_size=1)
        self.activation = F.relu if activation == 'relu' else F.gelu
        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = nn.BatchNorm1d(d_ff)

    def forward(self, x):
        '''
        x : N x L x C(=d_model)
        '''
        # out = self.decoder_layer1(x.transpose(-1, 1))
        # out = self.dropout(self.activation(self.batchnorm(out)))

        # decoder ablation
        # for _ in range(10):
        #     out = self.dropout(self.activation(self.decoder_layer_add(out)))

        # out = self.decoder_layer2(out).transpose(-1, 1)     
        '''
        out : reconstructed output
        '''
        out = self.out_linear(x)
        return out      # N x L x c_out


class TransformerVar(nn.Module):
    # ours: shrink_thres=0.0025
    def __init__(self, win_size, enc_in, c_out, n_memory, shrink_thres=0,\
                 d_model=512, n_heads=8, e_layers=3, d_ff=512, dropout=0.0, activation='gelu', \
                 device=None, memory_init_embedding=None, memory_initial=False, phase_type=None, dataset_name=None):
        super(TransformerVar, self).__init__()

        self.memory_initial = memory_initial

        # Encoding
        self.embedding = InputEmbedding(in_dim=enc_in, d_model=d_model, dropout=dropout, device=device)   # N x L x C(=d_model)
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        win_size, d_model, n_heads, dropout=dropout
                    ), d_model, d_ff, dropout=dropout, activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer = nn.LayerNorm(d_model)
        )

        self.mem_module = MemoryModule(n_memory=n_memory, fea_dim=d_model, shrink_thres=shrink_thres, device=device, memory_init_embedding=memory_init_embedding, phase_type=phase_type, dataset_name=dataset_name)
        

        #ours
        self.weak_decoder = Decoder(2*d_model, c_out, d_ff=d_ff, activation='gelu', dropout=0.1)

        # baselines
        # self.weak_decoder = Decoder(d_model, c_out, d_ff=d_ff, activation='gelu', dropout=0.1)


    def forward(self, x):
        '''
        x (input time window) : N x L x enc_in
        '''
        x = self.embedding(x)   # embeddin : N x L x C(=d_model)
        queries = out = self.encoder(x)   # encoder out : N x L x C(=d_model)
        
        outputs = self.mem_module(out)
        out, attn, memory_item_embedding = outputs['output'], outputs['attn'], outputs['memory_init_embedding']
        # out B T D
        # q B T D
        # mem  M D
        # memory_item_embedding M D
        # attn B T M

        mem = self.mem_module.mem
        
        if self.memory_initial:
            return {"out": out, "memory_item_embedding": None, "queries": queries, "mem": mem}
        else:
            
            out = self.weak_decoder(out)
            
            '''
            out (reconstructed input time window) : N x L x enc_in
            enc_in == c_out
            '''
            return {"out": out, "memory_item_embedding": memory_item_embedding, "queries": queries, "mem": mem, "attn": attn}


class TransformerANVar(nn.Module):
    def __init__(self, win_size, enc_in, c_out, n_memory, a_memory,factor,beta,cut_len, mask_ratio,noise_ratio,shrink_thres=0, \
                 d_model=512, n_heads=8, e_layers=3, d_ff=512, dropout=0.0, activation='gelu', \
                 device=None, memory_init_embedding=None, memory_init_embedding_a=None, memory_initial=False, phase_type= None, dataset_name=None):
        super(TransformerANVar, self).__init__()
        self.win_size = win_size
        self.mask_ratio = mask_ratio
        self.noise_ratio = noise_ratio
        self.memory_initial = memory_initial
        self.beta = beta
        self.cut_len = cut_len
        self.factor = factor
        # Encoding
        self.embedding = InputEmbedding(in_dim=enc_in, d_model=d_model, dropout=dropout, device=device)
        self.embedding_a = InputEmbedding(in_dim=enc_in, d_model=d_model, dropout=dropout, device=device)

        self.triplet = nn.TripletMarginLoss(margin=1)

        self.Amemory = Memory_Unit(nums=a_memory, dim=512, device=device)
        self.Nmemory = Memory_Unit(nums=n_memory, dim=512, device=device)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        win_size, d_model, n_heads, dropout=dropout
                    ), d_model, d_ff, dropout=dropout, activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        self.encoder_a = Encoder_a(
            [
                EncoderLayer(
                    AttentionLayer(
                        win_size, d_model, n_heads, dropout=dropout
                    ), d_model, d_ff, dropout=dropout, activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        self.mem_module = MemoryModule(n_memory=n_memory, fea_dim=d_model, shrink_thres=shrink_thres, device=device,
                                       memory_init_embedding=memory_init_embedding, phase_type=phase_type,
                                       dataset_name=dataset_name)

        self.mem_module_a = AMemoryModule(a_memory=a_memory, fea_dim=d_model, shrink_thres=shrink_thres, device=device,
                                       memory_init_embedding_a=memory_init_embedding_a, phase_type=phase_type,
                                       dataset_name=dataset_name)
        self.criterion = nn.MSELoss()
        self.pro = nn.Sequential(
            nn.Linear(2*d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.pro_a = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.encoder_mu = nn.Sequential(nn.Linear(d_model, d_model))
        self.encoder_var = nn.Sequential(nn.Linear(d_model, d_model))
        self.encoder_pro = nn.Sequential(nn.Linear(2*d_model, d_model))
        # Triplet loss for pseudo anomaly
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0)

        # ours
        # self.weak_decoder = Decoder(d_model, c_out, d_ff=d_ff, activation='gelu', dropout=0.1)
        # self.weak_decoder_a = Decoder(d_model, c_out, d_ff=d_ff, activation='gelu', dropout=0.1)

        self.weak_decoder = Decoder(2*d_model, c_out, d_ff=d_ff, activation='gelu', dropout=0.1)
        self.weak_decoder_a = Decoder(2*d_model, c_out, d_ff=d_ff, activation='gelu', dropout=0.1)

    def latent_loss(self, mu, var):
        kl_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim=1))
        return kl_loss

    def _reparameterize(self, mu, logvar):
        std = torch.exp(logvar).sqrt()
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def generate_pseudo_anomalies(self, x):
        """
        This function generates pseudo anomalies by adding noise or perturbations to the input x.
        """

        mean = x.mean(dim=1, keepdim=True)  # 计算每个特征的均值，结果形状为 (B, 1, C)
        std = x.std(dim=1, keepdim=True)  # 计算每个特征的标准差，结果形状为 (B, 1, C)
        # 使用均值和标准差生成噪声
        # 生成的噪声与输入的形状相同
        noise = torch.normal(mean=mean, std=std)
        noise = noise.to('cuda')
        factor = self.factor
        x = adjust_amplitude_in_frequency_domain(x, factor)
        pseudo_anomalies = x #+ noise

        # cut_length = self.win_size // 5
        cut_len = self.cut_len
        cut_length = int(cut_len * self.win_size)
        paste_position = random.randint(0, self.win_size - cut_length)
        b, t, d = x.size()

        trends = []

        for i in range(d):
            # 随机选择趋势的方向：1 表示上升，-1 表示下降
            direction = random.choice([1, -1])

            # 随机选择趋势的大小，大小可以是一个任意的幅度，假设在 [0.5, 5] 范围内
            magnitude = random.uniform(0.5, 5.0)

            # 生成趋势，基于方向和大小
            trend = torch.linspace(0, magnitude, cut_length) * direction
            trends.append(trend)

        # 将所有趋势堆叠成一个 (d, cut_length) 的张量
        trends = torch.stack(trends)
        trends = trends.transpose(1, 0)

        pseudo_anomalies = cut_add_paste(x, pseudo_anomalies, trends, cut_length, paste_position)

        return pseudo_anomalies, paste_position, cut_length


    def forward(self, x):
        '''
        x (input time window) : N x L x enc_in
        '''
        ########缺失值实验########
        # x = add_random_mask(x, self.win_size, self.mask_ratio)
        ######利用掩码实现########

        ########加噪声实验########
        # 添加高斯噪声
        # x = add_gaussian_noise(x, self.noise_ratio, noise_std=0.5)
        # 添加伽马噪声
        # x = add_gamma_noise(x, noise_ratio=0.2, concentration=2.0, rate=1.0)
        # 添加瑞利噪声
        # x = add_rayleigh_noise(x, noise_ratio=0.2, scale=1.0)
        # 添加指数噪声
        # x = add_exponential_noise(x, noise_ratio=0.2, rate=1.0)
        # 添加均匀噪声
        # x = add_uniform_noise(x, noise_ratio=0.2, low=-0.5, high=0.5)

        ax, pos, len = self.generate_pseudo_anomalies(x)
        out_ax = ax

        x = self.embedding(x)  # embeddin : N x L x C(=d_model)
        ax = self.embedding_a(ax)

        ######DualMemory########
        queries = out = self.encoder(x)
        queries_a = out_a = self.encoder_a(ax)

        outputs = self.mem_module(out)
        outputs_a = self.mem_module_a(out_a)

        #outputs_a = self.mem_module_an(out, out_a)

        ######正常记忆单元########
        out, attn, memory_item_embedding = outputs['output'], outputs['attn'], outputs['memory_init_embedding']
        mem = self.mem_module.mem
        ######异常记忆单元########
        out_a, attn_a, memory_item_embedding_a = outputs_a['output_a'], outputs_a['attn_a'], outputs_a['memory_init_embedding_a']
        mem_a = self.mem_module_a.mem_a

        out = self.pro(out)
        out_a = self.pro_a(out_a)

        # 特征表示
        # out_pa = out_a

        Z_out = out

        b, t, d = out.size()

        A_att, A_aug = self.Amemory(out_a, memory_item_embedding_a)
        N_Aatt, N_Aaug = self.Nmemory(out_a, memory_item_embedding)

        A_Natt, A_Naug = self.Amemory(out, memory_item_embedding_a)
        N_att, N_aug = self.Nmemory(out, memory_item_embedding)

        Z_aug = N_aug

        # # 从异常数据的异常注意力中提取负样本
        # _, A_index = torch.topk(A_att, t // 16 + 1, dim=-1)
        # negative_ax = torch.gather(out_a, 1, A_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1)
        # negative_ax = torch.gather(A_aug, 1, A_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1)


        # # 从正常数据的正常注意力中提取锚点
        # _, N_index = torch.topk(N_att, t // 16 + 1, dim=-1)
        # anchor_nx = torch.gather(out, 1, N_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1)
        # anchor_nx = torch.gather(N_aug, 1, N_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1)


        # # 从异常数据的正常注意力中提取伪正样本
        # _, P_index = torch.topk(N_Aatt, t // 16 + 1, dim=-1)
        # positivte_nx = torch.gather(out_a, 1, P_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1)
        # positivte_nx = torch.gather(N_Aaug, 1, P_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1)

        # 从正常数据的异常注意力中提取伪正样本
        #_, P_index = torch.topk(A_Natt, t // 16 + 1, dim=-1)
        # positivte_nx = torch.gather(out_a, 1, P_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1)
        # positivte_nx = torch.gather(A_Naug, 1, P_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1)


        # 计算三元组损失
        # con_loss = self.triplet(norm(anchor_nx), norm(positivte_nx), norm(negative_ax))
        # con_loss = instance_contrastive_loss(N_aug, N_Aaug, A_aug, A_Naug) + temporal_contrastive_loss(N_aug, A_aug)

        # mask = torch.zeros((b, t), dtype=torch.bool)
        # mask[:, pos:pos + len] = True
        # # # 将掩码应用于 A_aug
        # # # 保留 pos:pos+len 的片段，其余位置置零
        # A_masked = A_aug * mask.unsqueeze(-1).cuda()


        # con_loss = temporal_contrastive_loss(N_aug, A_masked) + instance_contrastive_loss(N_aug, N_Aaug, A_aug, A_Naug)

        con_loss = instance_contrastive_loss(N_aug, N_Aaug, A_aug, A_Naug) + temporal_contrastive_loss(N_aug, A_aug) #+
        # 正负样本的选择
        # 正负样本的个数

        # # VAE重构过程
        # N_aug_mu = self.encoder_mu(N_aug)

        # N_aug_var = self.encoder_var(N_aug)
        # N_aug_new = self._reparameterize(N_aug_mu, N_aug_var)

        # anchor_nx_new = torch.gather(N_aug_new, 1, N_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1)

        # A_aug_new = self.encoder_mu(A_aug)
        # negative_ax_new = torch.gather(A_aug_new, 1, A_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1)

        # # KL散度损失
        # kl_loss = self.latent_loss(N_aug_mu, N_aug_var)

        # # 计算正样本和负样本之间的距离
        # # A_Naug = self.encoder_mu(A_Naug)
        # # N_Aaug = self.encoder_mu(N_Aaug)
        # distance = torch.relu(1000 - torch.norm(negative_ax_new, p=2, dim=-1) + torch.norm(anchor_nx_new, p=2, dim=-1)).mean()

        # 使用编码器对增广数据进行推断
        N_aug_mu = self.encoder_mu(N_aug)  # 得到增广数据的均值
        N_aug_var = self.encoder_var(N_aug)  # 得到增广数据的方差

        # 进行重参数化，生成潜在空间的表示
        N_aug_new = self._reparameterize(N_aug_mu, N_aug_var)

        # 保留正负样本在潜在空间中的原始表示，而不是求均值
        # anchor_nx_new = torch.gather(N_aug_new, 1, N_index.unsqueeze(2).expand([-1, -1, x.size(-1)]))

        # 对负样本进行类似的处理
        A_aug_new = self.encoder_mu(A_aug)
        # negative_ax_new = torch.gather(A_aug_new, 1, A_index.unsqueeze(2).expand([-1, -1, x.size(-1)]))

        # 计算KL散度损失（对潜在变量的正则化）
        kl_loss = self.latent_loss(N_aug_mu, N_aug_var)

        # anchor_norm = torch.norm(anchor_nx_new, p=2, dim=-1)  # B x T
        # negative_norm = torch.norm(negative_ax_new, p=2, dim=-1)  # B x T
        # 计算正样本与负样本之间的L2距离
        # anchor_to_negative_distance = torch.norm(anchor_nx_new - negative_ax_new, p=2, dim=-1)  # B x T
        # 正样本之间的距离越小越好，负样本越远越好
        # 这里我们使用对比损失函数，正样本之间的距离应该小，负样本之间的距离应该大
        # positive_loss = torch.mean(torch.norm(anchor_nx_new, p=2, dim=-1))  # 聚集正样本

        distance = (1 - self.beta) * torch.mean(torch.relu(1 - torch.norm(N_aug_new - A_aug_new, p=2, dim=-1))) + self.beta * torch.mean(torch.norm(N_aug_new, p=2, dim=-1))  # 聚集正样本
        # distance = torch.mean(torch.relu(10-torch.norm(anchor_nx_new - negative_ax_new, p=2, dim=-1))) + torch.mean(torch.norm(anchor_nx_new, p=2, dim=-1)) # 聚集正样本

        # 计算正样本和负样本之间的距离
        #distance = torch.relu(100 - torch.norm(negative_ax_new, p=2, dim=-1) + torch.norm(anchor_nx_new, p=2, dim=-1)).mean()

        Z_con = N_aug
        Z_vae = N_aug
        Z_ul = N_aug
        # 距离大小

        # 拼接并通过分类头
        # x = torch.cat((out, self.encoder_pro(torch.cat([N_aug_new + A_Naug, A_aug_new + N_Aaug], dim=-1))), dim=-1)
        # x = torch.cat((out, self.encoder_pro(torch.cat([N_aug_new, A_aug_new], dim=-1))), dim=-1)

        # x = out + 0.5*N_aug_new
        # ax = out_a + 0.5*A_aug_new
        x = torch.cat((out, N_aug_new), dim=-1)
        # Concat N_aug
        ax = torch.cat((out_a, A_aug_new), dim=-1)

        out = x
        out_a = ax

        if self.memory_initial:
            return {"out": out, "memory_item_embedding": None, "memory_item_embedding_a": None, "queries": queries, "queries_a": queries_a,"mem": mem, 'triplet_margin': con_loss,
                    'kl_loss': kl_loss, 'distance': distance, 'Z_out': Z_out, 'Z_aug': Z_aug, 'Z_con': Z_con, 'Z_vae': Z_vae, 'Z_ul': Z_ul }
        else:
            out = self.weak_decoder(out)
            out_a = self.weak_decoder_a(out_a)

            '''
            out (reconstructed input time window) : N x L x enc_in
            enc_in == c_out
            '''
            return {"out": out, "memory_item_embedding": memory_item_embedding, "memory_item_embedding_a": memory_item_embedding_a, "mem_a": mem_a, "queries": queries,"queries_a": queries_a, "mem": mem,
                    "attn": attn, "attn_a": attn_a, 'triplet_margin': con_loss,
                    'kl_loss': kl_loss,
                    'distance': distance, 'Z_out': Z_out, 'Z_aug': Z_aug, 'Z_con': Z_con, 'Z_vae': Z_vae, 'Z_ul':N_aug_new, "A_att": A_att, 'N_Aatt': N_Aatt,
                    'A_Natt': A_Natt,
                    'N_att': N_att, 'out_a': out_a, 'out_pa': A_aug_new, 'out_ax': out_ax}
