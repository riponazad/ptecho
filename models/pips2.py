import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from utils.utils_ import bilinear_sample2d, posemb_sincos_2d_xy, reduce_masked_mean, sequence_loss, bilinear_sampler
from models.blocks import Conv1dPad, ResidualBlock1d, ResidualBlock2d, CorrBlock

class Pips(nn.Module):
    def __init__(self, stride:int=8):
        super(Pips, self).__init__()

        self.stride:int = stride

        self.hidden_dim:int = 256
        #hdim:int = 256
        self.latent_dim:int = 128
        #latent_dim:int = 128
        self.corr_levels:int = 4
        self.corr_radius:int = 3
        
        self.fnet = BasicEncoder(output_dim=self.latent_dim, norm_fn='instance', dropout=0, stride=stride)
        self.delta_block = DeltaBlock(hidden_dim=self.hidden_dim, corr_levels=self.corr_levels, corr_radius=self.corr_radius)
        self.norm = nn.GroupNorm(1, self.latent_dim)

    def forward(
            self,
            trajs_e0:torch.Tensor,
            rgbs:torch.Tensor,
            iters:int=3, 
            trajs_g:torch.Tensor=torch.empty(0),
            vis_g:torch.Tensor=torch.empty(0),
            valids:torch.Tensor=torch.empty(0),
            feat_init:None=None,
            is_train:bool=False, 
            beautify:bool=False,
            q_frame:int=0
        ):
        #total_loss = torch.tensor(0.0).cuda()

        B,S,N,D = trajs_e0.shape
        assert (D==2), "Dimension should be 2"
        

        B,S,C,H,W = rgbs.shape
        rgbs = 2 * (rgbs / 255.0) - 1.0
        
        H8 = H//self.stride
        W8 = W//self.stride

        #device = rgbs.device

        rgbs_ = rgbs.reshape(B*S, C, H, W)
        fmaps_ = self.fnet(rgbs_)
        fmaps = fmaps_.reshape(B, S, self.latent_dim, H8, W8)


        coords = trajs_e0.clone()/float(self.stride)

        #hdim = self.hidden_dim

        fcorr_fn1 = CorrBlock(fmaps, num_levels=self.corr_levels, radius=self.corr_radius)
        fcorr_fn2 = CorrBlock(fmaps, num_levels=self.corr_levels, radius=self.corr_radius)
        fcorr_fn4 = CorrBlock(fmaps, num_levels=self.corr_levels, radius=self.corr_radius)
        if feat_init is not None:
            feats1, feats2, feats4 = feat_init
        else:
            feat1 = bilinear_sample2d(fmaps[:,q_frame], coords[:,q_frame,:,0], coords[:,q_frame,:,1]).permute(0, 2, 1) # B,N,C
            feats1 = feat1.unsqueeze(1).repeat(1, S, 1, 1) # B,S,N,C
            feats2 = feat1.unsqueeze(1).repeat(1, S, 1, 1) # B,S,N,C
            feats4 = feat1.unsqueeze(1).repeat(1, S, 1, 1) # B,S,N,C
        
        coords_bak = coords.clone()

        coords[:,0] = coords_bak[:,0] # lock coord0 for target
        
        coord_predictions1 = [] # for loss
        coord_predictions2 = [] # for vis

        fcorr_fn1.corr(feats1) # we only need to run this corr once

        coord_predictions2.append(coords.detach() * self.stride)
        
        for itr in range(iters):
            coords = coords.detach()

            if itr >= 1:
                # timestep indices
                inds2 = (torch.arange(S)-2).clip(min=0)
                inds4 = (torch.arange(S)-4).clip(min=0)
                # coordinates at these timesteps
                coords2_ = coords[:,inds2].reshape(B*S,N,2)
                coords4_ = coords[:,inds4].reshape(B*S,N,2)
                # featuremaps at these timesteps
                fmaps2_ = fmaps[:,inds2].reshape(B*S,self.latent_dim,H8,W8)
                fmaps4_ = fmaps[:,inds4].reshape(B*S,self.latent_dim,H8,W8)
                # features at these coords/times
                feats2_ = bilinear_sample2d(fmaps2_, coords2_[:,:,0], coords2_[:,:,1]).permute(0, 2, 1) # B*S, N, C
                feats2 = feats2_.reshape(B,S,N,self.latent_dim)
                feats4_ = bilinear_sample2d(fmaps4_, coords4_[:,:,0], coords4_[:,:,1]).permute(0, 2, 1) # B*S, N, C
                feats4 = feats4_.reshape(B,S,N,self.latent_dim)

            fcorr_fn2.corr(feats2)
            fcorr_fn4.corr(feats4)

            # now we want costs at the current locations
            fcorrs1 = fcorr_fn1.sample(coords) # B,S,N,LRR
            fcorrs2 = fcorr_fn2.sample(coords) # B,S,N,LRR
            fcorrs4 = fcorr_fn4.sample(coords) # B,S,N,LRR
            LRR = fcorrs1.shape[3]

            # we want everything in the format B*N, S, C
            fcorrs1_ = fcorrs1.permute(0, 2, 1, 3).reshape(B*N, S, LRR)
            fcorrs2_ = fcorrs2.permute(0, 2, 1, 3).reshape(B*N, S, LRR)
            fcorrs4_ = fcorrs4.permute(0, 2, 1, 3).reshape(B*N, S, LRR)
            fcorrs_ = torch.cat([fcorrs1_, fcorrs2_, fcorrs4_], dim=2)
            flows_ = (coords[:,1:] - coords[:,:-1]).permute(0,2,1,3).reshape(B*N, S-1, 2)
            flows_ = torch.cat([flows_, flows_[:,-1:]], dim=1) # B*N,S,2

            delta_coords_ = self.delta_block(fcorrs_, flows_) # B*N,S,2

            if beautify and itr > 3*iters//4:
                # this smooths the results a bit, but does not really help perf
                delta_coords_ = delta_coords_ * 0.5

            coords = coords + delta_coords_.reshape(B, N, S, 2).permute(0,2,1,3)

            coord_predictions1.append(coords * self.stride)

            coords[:,0] = coords_bak[:,0] # lock coord0 for target
            coord_predictions2.append(coords * self.stride)
            
        # pause at the end, to make the summs more interpretable
        coord_predictions2.append(coords * self.stride)

        loss: float = 0.0
        if is_train:
            loss = sequence_loss(coord_predictions1, trajs_g, vis_g, valids, 0.8)
        else:
            loss = -1.0
            
        coord_predictions1.append(coords * self.stride)
        feats = (feats1, feats2, feats4)
        return coord_predictions1, coord_predictions2, feats, loss

    
class BasicEncoder(nn.Module):
    def __init__(self, input_dim: int = 3, output_dim: int = 128, stride: int = 8, norm_fn: str = 'batch', dropout: float = 0.0):
        super(BasicEncoder, self).__init__()
        self.stride: int = stride
        self.norm_fn: str = norm_fn

        self.in_planes: int = 64
        
        # Initialize normalization layers
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=self.in_planes) if norm_fn == 'group' else \
                     nn.InstanceNorm2d(self.in_planes) if norm_fn in ('batch', 'instance') else None

        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=output_dim * 2) if norm_fn == 'group' else \
                     nn.InstanceNorm2d(output_dim * 2) if norm_fn in ('batch', 'instance') else None
            
        # Conv and relu layers
        self.conv1 = nn.Conv2d(input_dim, self.in_planes, kernel_size=7, stride=2, padding=3, padding_mode='zeros')
        self.relu1 = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)
        self.layer4 = self._make_layer(128, stride=2)

        self.conv2 = nn.Conv2d(128+128+96+64, output_dim*2, kernel_size=3, padding=1, padding_mode='zeros')
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_dim*2, output_dim, kernel_size=1)
        
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim: int, stride: int = 1) -> nn.Sequential:
        layers = nn.ModuleList([
            ResidualBlock2d(self.in_planes, dim, self.norm_fn, stride=stride),
            ResidualBlock2d(dim, dim, self.norm_fn, stride=1)
        ])
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        x = self.conv1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu1(x)

        a = self.layer1(x)
        b = self.layer2(a)
        c = self.layer3(b)
        d = self.layer4(c)

        a = F.interpolate(a, (H // self.stride, W // self.stride), mode='bilinear', align_corners=True)
        b = F.interpolate(b, (H // self.stride, W // self.stride), mode='bilinear', align_corners=True)
        c = F.interpolate(c, (H // self.stride, W // self.stride), mode='bilinear', align_corners=True)
        d = F.interpolate(d, (H // self.stride, W // self.stride), mode='bilinear', align_corners=True)

        x = self.conv2(torch.cat([a, b, c, d], dim=1))
        if self.norm2 is not None:
            x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        return x

class DeltaBlock(nn.Module):
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 128, corr_levels: int = 4, corr_radius: int = 3):
        super(DeltaBlock, self).__init__()
        
        kitchen_dim = (corr_levels * (2*corr_radius + 1)**2)*3 + latent_dim + 2

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        in_channels = kitchen_dim
        base_filters = 128
        self.n_block = 8
        self.kernel_size = 3
        self.groups = 1
        self.use_norm = True
        self.use_do = False

        self.increasefilter_gap = 2 

        self.first_block_conv = Conv1dPad(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_norm = nn.InstanceNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters
                
        self.basicblock_list = nn.ModuleList()

        for i_block in range(self.n_block):
            is_first_block = i_block == 0
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                in_channels = int(base_filters * 2 ** ((i_block - 1) // self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            
            tmp_block = ResidualBlock1d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride=1, 
                groups=self.groups, 
                use_norm=self.use_norm, 
                use_do=self.use_do, 
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        self.final_norm = nn.InstanceNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        self.dense = nn.Linear(out_channels, 2)
        
            
    def forward(self, fcorr: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        B, S, D = flow.shape
        assert(D==2)
        flow_sincos = posemb_sincos_2d_xy(flow, self.latent_dim, cat_coords=True)
        x = torch.cat([fcorr, flow_sincos], dim=2) # B,S,-1
        
        # conv1d wants channels in the middle
        out = x.permute(0,2,1)
        out = self.first_block_conv(out)
        out = self.first_block_relu(out)
        for i, block in enumerate(self.basicblock_list):
            out = block(out)
        out = self.final_relu(out)
        out = out.permute(0,2,1)
        
        delta = self.dense(out)
        return delta


