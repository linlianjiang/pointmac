#! /usr/bin/python3
# -*- coding: utf-8 -*-
#
# @Last Modified by: Linlian Jiang
# @Date: 2025-12-08


import os
import torch
import copy
import math
import numpy as np
from torch import nn, einsum
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation, ball_query, three_nn, three_interpolate, grouping_operation
import torch.nn.functional as F
from torch import einsum
from models.utils import PointNet_SA_Module_KNN, MLP_Res, MLP_CONV, fps_subsample, query_knn, grouping_operation, get_nearest_index, indexing_neighbor

# VA from Point Transformer
class VectorAttention(nn.Module):
    def __init__(self, in_channel = 128, dim = 64, n_knn = 16, attn_hidden_multiplier = 4):
        super().__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)
        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1)
        )
        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )
        self.conv_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, query, support):
        pq, fq = query
        ps, fs = support

        identity = fq 
        query, key, value = self.conv_query(fq), self.conv_key(fs), self.conv_value(fs) 
        
        B, D, N = query.shape

        pos_flipped_1 = ps.permute(0, 2, 1).contiguous() 
        pos_flipped_2 = pq.permute(0, 2, 1).contiguous() 
        idx_knn = query_knn(self.n_knn, pos_flipped_1, pos_flipped_2)

        key = grouping_operation(key, idx_knn) 
        qk_rel = query.reshape((B, -1, N, 1)) - key  

        pos_rel = pq.reshape((B, -1, N, 1)) - grouping_operation(ps, idx_knn)  
        pos_embedding = self.pos_mlp(pos_rel) 

        attention = self.attn_mlp(qk_rel + pos_embedding) 
        attention = torch.softmax(attention, -1)

        value = grouping_operation(value, idx_knn) + pos_embedding  
        agg = einsum('b c i j, b c i j -> b c i', attention, value)  
        output = self.conv_end(agg) + identity
        
        return output

def hierarchical_fps(pts, rates):
    pts_flipped = pts.permute(0, 2, 1).contiguous()
    B, _, N = pts.shape
    now = N
    fps_idxs = []
    for i in range(len(rates)):
        now = now // rates[i]
        if now == N:
            fps_idxs.append(None)
        else:
            fps_idxs.append(furthest_point_sample(pts_flipped, now))
    return fps_idxs

# project f from p1 onto p2
def three_inter(f, p1, p2):
    # print(f.shape, p1.shape, p2.shape)
    # p1_flipped = p1.permute(0, 2, 1).contiguous()
    # p2_flipped = p2.permute(0, 2, 1).contiguous()
    idx, dis = get_nearest_index(p2, p1, k=3, return_dis=True) 
    dist_recip = 1.0 / (dis + 1e-8)
    norm = torch.sum(dist_recip, dim = 2, keepdim = True) 
    weight = dist_recip / norm
    proj_f = torch.sum(indexing_neighbor(f, idx) * weight.unsqueeze(1), dim=-1)
    return proj_f

# Cross-Resolution Transformer
class CRT(nn.Module):
    def __init__(self, dim_in = 128, is_inter = True, down_rates = [1, 4, 2], knns = [16, 12, 8]):
        super().__init__()
        self.down_rates = down_rates
        self.is_inter = is_inter
        self.num_scale = len(down_rates)

        self.attn_lists = nn.ModuleList()
        self.q_mlp_lists = nn.ModuleList()
        self.s_mlp_lists = nn.ModuleList()
        for i in range(self.num_scale):
            self.attn_lists.append(VectorAttention(in_channel = dim_in, dim = 64, n_knn = knns[i]))

        for i in range(self.num_scale - 1):
            self.q_mlp_lists.append(MLP_Res(in_dim = 128*2, hidden_dim = 128, out_dim = 128))
            self.s_mlp_lists.append(MLP_Res(in_dim = 128*2, hidden_dim = 128, out_dim = 128))

    def forward(self, query, support, fps_idxs_q = None, fps_idxs_s = None):
        pq, fq = query
        ps, fs = support
        # prepare fps_idxs_q and fps_idxs_s
        if fps_idxs_q == None:
            fps_idxs_q = hierarchical_fps(pq, self.down_rates)
        
        if fps_idxs_s == None:
            if self.is_inter:
                fps_idxs_s = hierarchical_fps(ps, self.down_rates) # inter-level
            else:
                fps_idxs_s = fps_idxs_q # intra-level
        
        # top-down aggregation
        pre_f = None
        pre_pos = None
        
        for i in range(self.num_scale - 1, -1, -1):
            if fps_idxs_q[i] == None:
                _pos1 = pq
            else:
                _pos1 = gather_operation(pq, fps_idxs_q[i])
            
            if fps_idxs_s[i] == None:
                _pos2 = ps
            else:
                _pos2 = gather_operation(ps, fps_idxs_s[i])

            if i == self.num_scale - 1:
                if fps_idxs_q[i] == None:
                    _f1 = fq
                else:
                    _f1 = gather_operation(fq, fps_idxs_q[i])
                if fps_idxs_s[i] == None:
                    _f2 = fs
                else:
                    _f2 = gather_operation(fs, fps_idxs_s[i])   
                
            else: 
                proj_f1 = three_inter(pre_f, pre_pos, _pos1)
                proj_f2 = three_inter(pre_f, pre_pos, _pos2)
                if fps_idxs_q[i] == None:
                    _f1 = fq
                else:
                    _f1 = gather_operation(fq, fps_idxs_q[i])
                if fps_idxs_s[i] == None:
                    _f2 = fs
                else:
                    _f2 = gather_operation(fs, fps_idxs_s[i]) 
                
                _f1 = self.q_mlp_lists[i](torch.cat([_f1, proj_f1], dim = 1))
                _f2 = self.s_mlp_lists[i](torch.cat([_f2, proj_f2], dim = 1))

            f = self.attn_lists[i]([_pos1, _f1], [_pos2, _f2])

            pre_f = f
            pre_pos = _pos1
        
        agg_f = pre_f
        return agg_f, fps_idxs_q, fps_idxs_s

# encoder
class Encoder(nn.Module):
    def __init__(self, out_dim = 512, n_knn = 16):
        super().__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all = False, if_bn = False, if_idx = True)
        self.crt_1 = CRT(dim_in = 128, is_inter = False, down_rates = [1, 2, 2], knns = [16, 12, 8])
        
        self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 128, [128, 256], group_all = False, if_bn = False, if_idx = True)
        self.conv_21 = nn.Conv1d(256, 128, 1)
        self.crt_2 = CRT(dim_in = 128, is_inter = False, down_rates = [1, 2, 2], knns = [16, 12, 8])
        self.conv_22 = nn.Conv1d(128, 256, 1)

        self.sa_module_3 = PointNet_SA_Module_KNN(None, None, 256, [512, out_dim], group_all = True, if_bn = False)

    def forward(self, partial_cloud):
        l0_xyz = partial_cloud
        l0_points = partial_cloud

        l1_xyz, l1_points, _ = self.sa_module_1(l0_xyz, l0_points)  
        l1_points, _, _ = self.crt_1([l1_xyz, l1_points], [l1_xyz, l1_points], None, None)

        l2_xyz, l2_points, _ = self.sa_module_2(l1_xyz, l1_points)
        l2_points_dim128 = self.conv_21(l2_points)
        l2_points_dim128, _, _ = self.crt_2([l2_xyz, l2_points_dim128], [l2_xyz, l2_points_dim128], None, None)
        l2_points = self.conv_22(l2_points_dim128) + l2_points

        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  

        return l2_xyz, l2_points, l3_points

class UpTransformer(nn.Module):
    def __init__(self, in_channel=128, out_channel=128, dim=64, n_knn=20, up_factor=2,
                 pos_hidden_dim=64, attn_hidden_multiplier=4, scale_layer=nn.Softmax, attn_channel=True):
        super(UpTransformer, self).__init__()
        self.n_knn = n_knn
        self.up_factor = up_factor
        attn_out_channel = dim if attn_channel else 1

        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)



        self.scale = scale_layer(dim=-1) if scale_layer is not None else nn.Identity()

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        # attention layers
        self.attn_mlp = [nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
                         nn.BatchNorm2d(dim * attn_hidden_multiplier),
                         nn.ReLU()]
        if up_factor:
            self.attn_mlp.append(nn.ConvTranspose2d(dim * attn_hidden_multiplier, attn_out_channel, (up_factor,1), (up_factor,1)))
        else:
            self.attn_mlp.append(nn.Conv2d(dim * attn_hidden_multiplier, attn_out_channel, 1))
        self.attn_mlp = nn.Sequential(*self.attn_mlp)

        # upsample previous feature
        self.upsample1 = nn.Upsample(scale_factor=(up_factor,1)) if up_factor else nn.Identity()
        self.upsample2 = nn.Upsample(scale_factor=up_factor) if up_factor else nn.Identity()

        # residual connection
        self.conv_end = nn.Conv1d(dim, out_channel, 1)
        if in_channel != out_channel:
            self.residual_layer = nn.Conv1d(in_channel, out_channel, 1)
        else:
            self.residual_layer = nn.Identity()

    def forward(self, pos1, query, pos2, key):
        """
        Inputs:
            pos: (B, 3, N)
            key: (B, in_channel, N)
            query: (B, in_channel, N)
        """

        value = key # (B, dim, N)
        identity = query
        key = self.conv_key(key) # (B, dim, N)
        query = self.conv_query(query)
        value = self.conv_value(value)
        b, dim, n = query.shape

        pos1_flipped = pos1.permute(0, 2, 1).contiguous()
        pos2_flipped = pos2.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos2_flipped, pos1_flipped) # b, N1, k

        key = grouping_operation(key, idx_knn)  # (B, dim, N1, k)
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos1.reshape((b, -1, n, 1)) - grouping_operation(pos2, idx_knn)  # (B, 3, N, k)
        pos_embedding = self.pos_mlp(pos_rel)  # (B, dim, N, k)

        # attention
        attention = self.attn_mlp(qk_rel + pos_embedding) # (B, dim, N*up_factor, k)

        # softmax function
        attention = self.scale(attention)

        # knn value is correct
        value = grouping_operation(value, idx_knn) + pos_embedding # (B, dim, N, k)
        value = self.upsample1(value) # (B, dim, N*up_factor, k)

        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # (B, dim, N*up_factor)
        y = self.conv_end(agg) # (B, out_dim, N*up_factor)

        # shortcut
        identity = self.residual_layer(identity) # (B, out_dim, N)
        identity = self.upsample2(identity) # (B, out_dim, N*up_factor)

        return y+identity

# seed generator using upsample transformer
class SeedGenerator(nn.Module):
    def __init__(self, feat_dim = 512, seed_dim = 128, n_knn = 16, factor = 2, attn_channel = True):
        super().__init__()
        self.uptrans = UpTransformer(in_channel = 256, out_channel = 128, dim = 64, n_knn = n_knn, attn_channel = attn_channel, up_factor = factor, scale_layer = None)
        self.mlp_1 = MLP_Res(in_dim = feat_dim + 128, hidden_dim = 128, out_dim = 128)
        self.mlp_2 = MLP_Res(in_dim = 128, hidden_dim = 64, out_dim = 128)
        self.mlp_3 = MLP_Res(in_dim = feat_dim + 128, hidden_dim = 128, out_dim = seed_dim)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(seed_dim, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat, patch_xyz, patch_feat, partial):
        x1 = self.uptrans(patch_xyz, patch_feat, patch_xyz, patch_feat)  # (B, 128, 256)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))  # (B, 128, 256)
        seed = self.mlp_4(x3)  # (B, 3, 256)
        x = fps_subsample(torch.cat([seed.permute(0, 2, 1).contiguous(), partial], dim=1), 512).permute(0, 2, 1).contiguous() # b, 3, 512
        return seed, x3, x

# seed generator using deconvolution
class SeedGenerator_Deconv(nn.Module):
    def __init__(self, feat_dim = 512, seed_dim = 128, n_knn = 16, factor = 2, attn_channel = True):
        super().__init__()
        num_pc = 256
        self.ps = nn.ConvTranspose1d(feat_dim, 128, num_pc, bias=True)
        self.mlp_1 = MLP_Res(in_dim = feat_dim + 128, hidden_dim = 128, out_dim = 128)
        self.mlp_2 = MLP_Res(in_dim = 128, hidden_dim = 64, out_dim = 128)
        self.mlp_3 = MLP_Res(in_dim = feat_dim + 128, hidden_dim = 128, out_dim = seed_dim)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(seed_dim, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat, patch_xyz, patch_feat, partial):
        x1 = self.ps(feat)  # (B, 128, 256)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))  # (B, 128, 256)
        seed = self.mlp_4(x3)  # (B, 3, 256)
        x = fps_subsample(torch.cat([seed.permute(0, 2, 1).contiguous(), partial], dim=1), 512).permute(0, 2, 1).contiguous() # b, 3, 512
        return seed, x3, x

# mini-pointnet
class PN(nn.Module):
    def __init__(self, feat_dim = 512):
        super().__init__()
        self.mlp_1 = MLP_CONV(in_channel = 3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel = 128 * 2 + feat_dim , layer_dims=[512, 256, 128])
        
    def forward(self, xyz, global_feat):
        b, _, n = xyz.shape
        feat = self.mlp_1(xyz)
        feat4cat = [feat, torch.max(feat, 2, keepdim=True)[0].repeat(1, 1, n), global_feat.repeat(1, 1, n)]
        point_feat = self.mlp_2(torch.cat(feat4cat, dim=1))
        return point_feat

class DeConv(nn.Module):
    def __init__(self, up_factor = 4):
        super().__init__()
        self.decrease_dim = MLP_CONV(in_channel = 128, layer_dims = [64, 32], bn = True)
        self.ps = nn.ConvTranspose1d(32, 128, up_factor, up_factor, bias = False)  
        self.mlp_res = MLP_Res(in_dim = 128 * 2, hidden_dim = 128, out_dim = 128)
        self.upper = nn.Upsample(scale_factor = up_factor)
        self.xyz_mlp = MLP_CONV(in_channel = 128, layer_dims = [64, 3])
    def forward(self, xyz, feat):
        feat_child = self.ps(self.decrease_dim(feat))
        feat_child = self.mlp_res(torch.cat([feat_child, self.upper(feat)], dim=1)) # b, 128, n*r
        delta = self.xyz_mlp(torch.relu(feat_child)) 
        new_xyz = self.upper(xyz) + torch.tanh(delta)
        return new_xyz


# upsampling block 
class UpBlock(nn.Module):
    def __init__(self, feat_dim = 512, down_rates = [1, 4, 2], knns = [16, 12, 8], up_factor = 4):
        super().__init__()
        self.pn = PN()
        self.inter_crt = CRT(dim_in = 128, is_inter = True, down_rates = down_rates, knns = knns)
        self.intra_crt = CRT(dim_in = 128, is_inter = False, down_rates = down_rates, knns = knns)
        self.deconv = DeConv(up_factor = up_factor)

    def forward(self, p, gf, pre, fps_idxs_1, fps_idxs_2):
        h = self.pn(p, gf)
        g, fps_idxs_q1, fps_idxs_s1 = self.inter_crt([p, h], pre, None, fps_idxs_1)
        f, _, _ = self.intra_crt([p, g], [p, g], fps_idxs_q1, fps_idxs_q1)
        new_xyz = self.deconv(p, f)
        return new_xyz, f, fps_idxs_q1, fps_idxs_s1

# decoder

class Decoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.ub0 = UpBlock(feat_dim = 512, down_rates = [1, 2, 2], knns = [16, 12, 8], up_factor = 1)
        self.ub1 = UpBlock(feat_dim = 512, down_rates = [1, 2, 2], knns = [16, 12, 8], up_factor = 2)
        self.ub2 = UpBlock(feat_dim = 512, down_rates = [1, 4, 2], knns = [16, 12, 8], up_factor = 8)

    def forward(self, global_f, p0, p_sd, f_sd):
        p1, f0, p0_fps_idxs_122, _ = self.ub0(p0, global_f, [p_sd, f_sd], None, None)
        p2, f1, p1_fps_idxs_122, _ = self.ub1(p1, global_f, [p0, f0], None, p0_fps_idxs_122)
        p3, _ , _______________, _ = self.ub2(p2, global_f, [p1, f1], None, None)
        
        all_pc = [p_sd.permute(0, 2, 1).contiguous(), p1.permute(0, 2, 1).contiguous(), \
            p2.permute(0, 2, 1).contiguous(), p3.permute(0, 2, 1).contiguous()]
        return all_pc

class Decoder_sn55(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.ub0 = UpBlock(feat_dim = 512, down_rates = [1, 2, 2], knns = [16, 12, 8], up_factor = 1)
        self.ub1 = UpBlock(feat_dim = 512, down_rates = [1, 2, 2], knns = [16, 12, 8], up_factor = 4)
        self.ub2 = UpBlock(feat_dim = 512, down_rates = [1, 4, 2], knns = [16, 12, 8], up_factor = 4)

    def forward(self, global_f, p0, p_sd, f_sd):
        p1, f0, p0_fps_idxs_122, _ = self.ub0(p0, global_f, [p_sd, f_sd], None, None)
        p2, f1, p1_fps_idxs_122, _ = self.ub1(p1, global_f, [p0, f0], None, p0_fps_idxs_122)
        p3, _ , _______________, _ = self.ub2(p2, global_f, [p1, f1], None, None)
        
        all_pc = [p_sd.permute(0, 2, 1).contiguous(), p1.permute(0, 2, 1).contiguous(), \
            p2.permute(0, 2, 1).contiguous(), p3.permute(0, 2, 1).contiguous()]
        return all_pc

class Decoder_mvp(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.ub0 = UpBlock(feat_dim = 512, down_rates = [1, 2, 2], knns = [16, 12, 8], up_factor = 1)
        self.ub1 = UpBlock(feat_dim = 512, down_rates = [1, 2, 2], knns = [16, 12, 8], up_factor = 4)

    def forward(self, global_f, p0, p_sd, f_sd):
        p1, f0, p0_fps_idxs_122, _ = self.ub0(p0, global_f, [p_sd, f_sd], None, None)
        p2, f1, _, __ = self.ub1(p1, global_f, [p0, f0], None, p0_fps_idxs_122)
        
        all_pc = [p_sd.permute(0, 2, 1).contiguous(), p1.permute(0, 2, 1).contiguous(), \
            p2.permute(0, 2, 1).contiguous()]
        return all_pc


# CRA-PCN 
class CRAPCN(nn.Module):
    def __init__(self, use_mae_aux=False, use_denoise_aux=False):
        """
        Args:
            use_mae_aux (bool): instantiate MAE auxiliary head (`ExtendedModel`) if True.
            use_denoise_aux (bool): instantiate denoising auxiliary head (`ExtendedModel2`) if True.
        """
        super().__init__()
        self.encoder = Encoder()
        self.seed_generator = SeedGenerator()
        self.decoder = Decoder()

        # Optional auxiliary heads (not used in current training scripts,
        # but kept for potential joint inference)
        self.use_mae_aux = use_mae_aux
        self.use_denoise_aux = use_denoise_aux

        self.mae_aux = ExtendedModel(self) if use_mae_aux else None
        # pass mae_aux for API compatibility, though it is not used internally
        self.denoise_aux = ExtendedModel2(self, self.mae_aux) if use_denoise_aux else None

    def forward(self, xyz, return_aux=False):
        """
        Args:
            xyz: (B, N, 3) partial / noisy input cloud.
            return_aux (bool): if True, also return outputs of enabled auxiliary heads.
        Returns:
            If return_aux is False:
                all_pc: List of point clouds from the main completion decoder.
            If return_aux is True:
                (all_pc, aux_outputs) where:
                    aux_outputs is a dict with optional keys:
                        'mae_rec': (B, 2048, 3) from `ExtendedModel` (MAE reconstruction)
                        'denoise_offset': (B, 1024, 3) from `ExtendedModel2` (offsets)
        """
        pp, fp, global_f = self.encoder(xyz.permute(0, 2, 1).contiguous())
        p_sd, f_sd, p0 = self.seed_generator(global_f, pp, fp, xyz)
        all_pc = self.decoder(global_f, p0, p_sd, f_sd)

        if not return_aux:
            return all_pc

        aux_outputs = {}
        if self.mae_aux is not None:
            aux_outputs['mae_rec'] = self.mae_aux(xyz)
        if self.denoise_aux is not None:
            aux_outputs['denoise_offset'] = self.denoise_aux(xyz)

        return all_pc, aux_outputs


## axuilty task ITSI + MAE
class DecoderFC(nn.Module):
    def __init__(self, n_features=(256, 256), latent_dim=128, output_pts=2048, bn=False):
        super(DecoderFC, self).__init__()
        self.n_features = list(n_features)
        self.output_pts = output_pts
        self.latent_dim = latent_dim

        model = []
        prev_nf = self.latent_dim
        for idx, nf in enumerate(self.n_features):
            fc_layer = nn.Linear(prev_nf, nf)
            model.append(fc_layer)

            if bn:
                bn_layer = nn.BatchNorm1d(nf)
                model.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf

        fc_layer = nn.Linear(self.n_features[-1], output_pts*3)
        model.append(fc_layer)

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        x = x.view((-1, 3, self.output_pts))
        return x


class ITSI(nn.Module):
    """
    Token Synergy Integrator (ITSI), shared across Bi-Aux Units.

    This module maps the shared encoder's global feature F (512-d) into:
      - a latent vector for stochastic masked reconstruction Auxsmr (128-d), and
      - a group-token matrix T_G for artifact denoising Auxad (64 x 1024 here).

    In the paper notation, ITSI carries the shared parameters ϕ_sh_aux that
    provide a unified conditioning space for both auxiliary branches.
    """
    def __init__(self, in_dim=512, latent_dim=128, token_channels=64, num_tokens=1024):
        super(ITSI, self).__init__()
        self.latent_fc = nn.Linear(in_dim, latent_dim)
        self.token_fc = nn.Sequential(
            nn.Linear(in_dim, token_channels * num_tokens),
            nn.BatchNorm1d(token_channels * num_tokens),
        )
        self.token_channels = token_channels
        self.num_tokens = num_tokens

    def forward(self, global_feat):
        """
        Args:
            global_feat: (B, in_dim) global feature F from the shared encoder E_sh.
        Returns:
            latent:  (B, latent_dim)           – Auxsmr latent vector (z_smr)
            tokens:  (B, token_channels, num_tokens) – group-token matrix T_G
        """
        latent = self.latent_fc(global_feat)
        tokens = self.token_fc(global_feat)
        tokens = F.relu(tokens).view(-1, self.token_channels, self.num_tokens)
        return latent, tokens


class ExtendedModel(nn.Module):
    """
    Auxsmr head: Stochastic Masked Reconstruction branch of the Bi-Aux Units.

    This branch:
      - reuses the shared encoder E_sh from the primary completion network,
      - applies stochastic masking on the input point cloud (MAE-style),
      - passes the resulting global feature F through ITSI (ϕ_sh_aux) to obtain
        a compact latent code z_smr, and
      - decodes z_smr with a lightweight MLP (DecoderFC) to reconstruct a dense
        point cloud P_e supervised by Chamfer Distance.
    """
    def __init__(self, original_model):
        super(ExtendedModel, self).__init__()

        self.feat_extractor_ex = original_model.encoder

        # masked autoencoder settings (Auxsmr)
        # ratio in [0, 1): fraction of input points to mask during training
        self.mask_ratio = 0.6

        #### newadd
        self.number_fine = 8192
        # Shared ITSI module (Token Synergy Integrator, ϕ_sh_aux)
        # used by both Auxsmr (this class) and Auxad (ExtendedModel2).
        self.itsi = ITSI(in_dim=512, latent_dim=128, token_channels=64, num_tokens=1024)
        ########## MAE Decoder
        self.decoder = DecoderFC(n_features=(256, 256), latent_dim=128, output_pts=2048, bn=False)
        ###Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _random_mask_points(self, x):
        """
        Stochastic masking of input points (Auxsmr).

        This implements the stochastic masked reconstruction strategy by
        randomly selecting a subset of visible points and dropping the rest,
        which regularizes the encoder against diverse missing patterns.
        Args:
            x: (B, N, 3) input point cloud
        Returns:
            visible_x: (B, N_keep, 3) visible points after masking
        """
        if self.mask_ratio <= 0 or not self.training:
            # no masking (e.g. evaluation) or disabled
            return x

        B, N, C = x.shape
        N_keep = max(1, int(N * (1.0 - self.mask_ratio)))

        # sample a different random subset for each batch element
        noise = torch.rand(B, N, device=x.device)
        ids_keep = torch.topk(noise, k=N_keep, dim=1, largest=False)[1]  # (B, N_keep)

        # gather points
        ids_keep_expanded = ids_keep.unsqueeze(-1).expand(-1, -1, C)      # (B, N_keep, 3)
        visible_x = torch.gather(x, dim=1, index=ids_keep_expanded)
        return visible_x

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        bs , n , _ = x.shape

        # Stochastic masking: only a subset of points is visible to the encoder during training
        x_visible = self._random_mask_points(x)  # (B, N_visible, 3) or full x at eval

        pc_x = x_visible.permute(0, 2, 1).contiguous()
        _, _, z3 = self.feat_extractor_ex(pc_x)

        global_feat = torch.max(z3, dim=2)[0]             # (B, 512)
        z_latent, _ = self.itsi(global_feat)              # (B, 128), (B, 64, 1024)

        x_rec = self.decoder(z_latent)                    # (B, 3, output_pts)
        return x_rec.transpose(1, 2).contiguous()         # (B, output_pts, 3)


## axuilty task 2: Point Cloud Denoising (offset prediction)
class SelfAttentionUnit(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionUnit, self).__init__()

        self.to_q = nn.Sequential(
            nn.Conv1d(3 + in_channels, 2 * in_channels, 1, bias=False),
            nn.BatchNorm1d(2 * in_channels),
            nn.ReLU(inplace=True)
        )
        self.to_k = nn.Sequential(
            nn.Conv1d(3 + in_channels, 2 * in_channels, 1, bias=False),
            nn.BatchNorm1d(2 * in_channels),
            nn.ReLU(inplace=True)
        )
        self.to_v = nn.Sequential(
            nn.Conv1d(3 + in_channels, 2 * in_channels, 1, bias=False),
            nn.BatchNorm1d(2 * in_channels),
            nn.ReLU(inplace=True)
        )

        self.fusion = nn.Sequential(
            nn.Conv1d(2 * in_channels, in_channels, 1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: (B, C + 3, 4N)
        Returns:
            (B, C, 4N)
        """
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        attention_map = torch.matmul(q.permute(0, 2, 1), k)          # (B, 4N, 4N)
        value = torch.matmul(attention_map, v.permute(0, 2, 1))      # (B, 4N, 2C)
        value = value.permute(0, 2, 1).contiguous()                  # (B, 2C, 4N)

        value = self.fusion(value)                                   # (B, C, 4N)
        return value


class OffsetRegression(nn.Module):
    def __init__(self, in_channels):
        super(OffsetRegression, self).__init__()
        self.coordinate_regression = nn.Sequential(
            nn.Conv1d(in_channels, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 3, 1),
            nn.Sigmoid()
        )
        self.range_max = 0.5

    def forward(self, x):
        """
        Args:
            x: (B, C, 4N)
        Returns:
            (B, 4N, 3)
        """
        offset = self.coordinate_regression(x)                       # (B, 3, 4N)
        offset = offset * self.range_max * 2 - self.range_max       # [-range_max, range_max]
        return offset.permute(0, 2, 1).contiguous()


class ExtendedModel2(nn.Module):
    """
    Auxad head: Artifact Denoising branch of the Bi-Aux Units.

    This auxiliary branch:
      - reuses the shared encoder E_sh to obtain a global feature F,
      - reuses the shared Token Synergy Integrator ITSI (ϕ_sh_aux) to obtain
        a group-token matrix T_G in a unified token space,
      - refines T_G with a self-attention unit to produce artifact-aware
        features R_G, and
      - predicts per-point geometric offsets Υ_ad (via OffsetRegression) to
        denoise and densify the input, supervised by Chamfer Distance.

    In code, the offsets are predicted on 1024 tokens, corresponding to the
    εM points described in the paper.
    """
    def __init__(self, priM, aux1M=None):
        """
        Args:
            priM: primary completion model (e.g. CRAPCN instance)
            aux1M: first auxiliary model (ExtendedModel), used to share ITSI
        """
        super(ExtendedModel2, self).__init__()

        # Reuse encoder from primary model (same as in ExtendedModel)
        self.feat_extractor_ex = priM.encoder

        # Shared ITSI from ExtendedModel (if provided), otherwise create a new one
        if aux1M is not None and hasattr(aux1M, "itsi"):
            self.itsi = aux1M.itsi
        else:
            self.itsi = ITSI(in_dim=512, latent_dim=128, token_channels=64, num_tokens=1024)

        # Self-attention and offset regression
        self.sa = SelfAttentionUnit(in_channels=61)
        self.offset = OffsetRegression(in_channels=61)

    def forward(self, pts):
        """
        Args:
            pts: (B, N, 3) noisy / artifact-corrupted input point cloud P
        Returns:
            refine: (B, 1024, 3) predicted offsets Υ_ad_ε(P) that are applied to
                    upsampled points to obtain the refined clean cloud P_b.
        """
        bs, n, _ = pts.shape

        # Ensure that all submodules live on the same device as the input tensor.
        # This makes the head robust even if the caller forgets to call `.cuda()`
        # / `.to(device)` on the whole `ExtendedModel2` instance.
        device = pts.device
        self.feat_extractor_ex = self.feat_extractor_ex.to(device)
        self.itsi = self.itsi.to(device)
        self.sa = self.sa.to(device)
        self.offset = self.offset.to(device)

        pc_x = pts.permute(0, 2, 1).contiguous()                    # (B, 3, N)
        _, _, global_f = self.feat_extractor_ex(pc_x)               # (B, 512, 1)
        global_feat = torch.max(global_f, dim=2)[0]                 # (B, 512)

        _, group_input_tokens = self.itsi(global_feat)              # (B, 64, 1024)
        t = self.sa(group_input_tokens)                             # (B, 61, 1024)
        refine = self.offset(t)                                     # (B, 1024, 3)

        return refine.contiguous()


# -------------------------------------------------------------------------
# Paper-aligned aliases (for readability w.r.t. the Bi-Aux description)
# -------------------------------------------------------------------------
# Auxsmr (ϕ_smr_aux): Stochastic Masked Reconstruction head
Auxsmr = ExtendedModel
# Auxad (ϕ_ad_aux): Artifact Denoising head
Auxad = ExtendedModel2
# ITSI / Token Synergy Integrator (ϕ_sh_aux)
TokenSynergyIntegrator = ITSI


class CRAPCN_sn55(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.encoder = Encoder()
        self.seed_generator = SeedGenerator()
        self.decoder = Decoder_sn55()

    def forward(self, xyz):
        pp, fp, global_f = self.encoder(xyz.permute(0, 2, 1).contiguous())
        p_sd, f_sd, p0 = self.seed_generator(global_f, pp, fp, xyz)
        all_pc = self.decoder(global_f, p0, p_sd, f_sd)
        return all_pc

class CRAPCN_mvp(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.encoder = Encoder()
        self.seed_generator = SeedGenerator()
        self.decoder = Decoder_mvp()

    def forward(self, xyz):
        pp, fp, global_f = self.encoder(xyz.permute(0, 2, 1).contiguous())
        p_sd, f_sd, p0 = self.seed_generator(global_f, pp, fp, xyz)
        all_pc = self.decoder(global_f, p0, p_sd, f_sd)
        return all_pc

# CRA-PCN 
class CRAPCN_d(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.encoder = Encoder()
        self.seed_generator = SeedGenerator_Deconv()
        self.decoder = Decoder()

    def forward(self, xyz):
        pp, fp, global_f = self.encoder(xyz.permute(0, 2, 1).contiguous())
        p_sd, f_sd, p0 = self.seed_generator(global_f, pp, fp, xyz)
        all_pc = self.decoder(global_f, p0, p_sd, f_sd)
        return all_pc

class CRAPCN_sn55_d(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.encoder = Encoder()
        self.seed_generator = SeedGenerator_Deconv()
        self.decoder = Decoder_sn55()

    def forward(self, xyz):
        pp, fp, global_f = self.encoder(xyz.permute(0, 2, 1).contiguous())
        p_sd, f_sd, p0 = self.seed_generator(global_f, pp, fp, xyz)
        all_pc = self.decoder(global_f, p0, p_sd, f_sd)
        return all_pc

class CRAPCN_mvp_d(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.encoder = Encoder()
        self.seed_generator = SeedGenerator_Deconv()
        self.decoder = Decoder_mvp()

    def forward(self, xyz):
        pp, fp, global_f = self.encoder(xyz.permute(0, 2, 1).contiguous())
        p_sd, f_sd, p0 = self.seed_generator(global_f, pp, fp, xyz)
        all_pc = self.decoder(global_f, p0, p_sd, f_sd)
        return all_pc

