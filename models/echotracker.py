""" 
Created on Thursday June 03 2025
@author: Azad Md Abulkalam
@location: ISB, NTNU

EchoTracker for tissue tracking in echocardiography videos.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import (
    CorrBlock, BasicEncoder, DeltaBlock, DeltaBlock_temp
)
from utils.utils_ import bilinear_sample2d
torch.manual_seed(0)
from typing import List


class EchoTracker(nn.Module):
    def __init__(
            self,
            stride: int=8, #spatial stride
            hidden_dim:int=128,
            latent_dim:int=64,
            corr_radius:int=3,
            corr_levels:int=4,
            scale:int = 4, # 
            feature_extractor_chunk_size:int = 64
    ):
        super(EchoTracker, self).__init__()
        self.stride:int = stride
        self.hidden_dim:int = hidden_dim
        self.latent_dim:int = latent_dim
        self.corr_levels:int = corr_levels
        self.corr_radius:int = corr_radius
        self.scale:int = scale
        self.feature_extractor_chunk_size:int = feature_extractor_chunk_size
        
        self.fnet = BasicEncoder(
            input_dim=1, output_dim=self.latent_dim, norm_fn="instance",
            dropout=0, stride=self.stride
        )
        self.fnet_fine = BasicEncoder(
            input_dim=1, output_dim=self.latent_dim, norm_fn="instance",
            dropout=0, stride=int(self.stride/self.scale)
        )
        self.delta_block_spatial = DeltaBlock(
            hidden_dim=self.hidden_dim, latent_dim=self.latent_dim,
            corr_levels=self.corr_levels, corr_radius=self.corr_radius
        )
        self.delta_block_temporal = DeltaBlock_temp(
            hidden_dim=self.hidden_dim, latent_dim=self.latent_dim,
            corr_levels=self.corr_levels, corr_radius=self.corr_radius
        )
        self.linear = nn.Linear(588, 196+196)
       
    def initialize_(self, rgbs:torch.Tensor, points_0:torch.Tensor, q_frame:int=0):
        """
        Initialize trajectories with batched data.

        Args:
            rgbs (torch.Tensor): Input frames of shape [B, S, C, H, W].
            points_0 (torch.Tensor): Initial points of shape [B, N, 2].
            q_frame (int): Index of the query frame for locking coordinates. Default is 0.

        Returns:
            torch.Tensor: Estimated trajectories of shape [B, S, N, 2].
        """
        B, S, C, H, W = rgbs.shape
        B, N, D = points_0.shape
        H_ = H // self.stride
        W_ = W // self.stride

        # Reshape and extract feature maps
        rgbs_ = rgbs.reshape(B * S, C, H, W)
        if self.feature_extractor_chunk_size > 0:
            fmaps_list: List[torch.Tensor] = []
            chunk_size = self.feature_extractor_chunk_size
            for start_idx in range(0, rgbs_.shape[0], chunk_size):
                rgb_chunk = rgbs_[start_idx:start_idx + chunk_size]
                fmaps_list.append(self.fnet(rgb_chunk))
            fmaps_ = torch.cat(fmaps_list, dim=0)
        else:
            fmaps_ = self.fnet(rgbs_)
        fmaps = fmaps_.reshape(B, S, self.latent_dim, H_, W_)
        rgbs_ = rgbs_.reshape(B, S, C, H, W)
        
        # Scale and replicate initial coordinates
        #print(points_0.shape, points_0.sum())
        coords_0 = points_0.detach() / float(self.stride)
        coords = coords_0.unsqueeze(1).repeat(1, S, 1, 1)  # [B, S, N, 2]
        

        # Extract feature vectors for points from the query frame
        f_vecs0 = bilinear_sample2d(fmaps[:, q_frame], coords[:, q_frame, :, 0], coords[:, q_frame, :, 1]).permute(0, 2, 1)  # [B, N, C]
        feats = f_vecs0.unsqueeze(1).repeat(1, S, 1, 1)  # [B, S, N, C]

        # Compute cost volume using pyramids
        fcorr_fn = CorrBlock(fmaps, num_levels=self.corr_levels, radius=self.corr_radius)
        
        coords_bak = coords.clone()
        
        # Generate flow features and prepare cross-correlation
        rgbs_ = rgbs_.squeeze(2)
        frame_flow = torch.cat([rgbs_[:,-1:], (rgbs_[:, 1:] - rgbs_[:, :-1])], dim=1)#.reshape(B*S, H, W)#.unsqueeze(1)
        frame_flow = F.interpolate(frame_flow, (50, 60), mode="bilinear").reshape(B, 1, S, 50 * 60).repeat(1, N, 1, 1).reshape(B*N, S, 50 * 60)
        
        # print(coords_bak.shape, coords_bak[:, q_frame].shape)
        # raise KeyboardInterrupt
        xys0 = coords_bak[:, q_frame].unsqueeze(2).repeat(1, 1, S, 1).reshape(B*N, S, 2)
        coords = coords.detach()
        fcorr_fn.corr(feats, chunk_size=self.feature_extractor_chunk_size)
        # for fm in fcorr_fn.corrs_pyramid:
        #     print(f"in initialize: {fm.shape}")
        
        # Process cross-correlations and interpolate for uniform size
        fcorrs = []
        for c_v in fcorr_fn.corrs_pyramid:
            B, S, N, CH, CW = c_v.shape
            c_v = c_v.reshape(B * S, N, CH, CW)
            c_v = F.interpolate(c_v, (14, 14), mode="bilinear", align_corners=True)
            fcorrs.append(c_v.reshape(B, S, N, 14 * 14))
        fcorrs = torch.stack(fcorrs).mean(dim=0)  # [B, S, N, 14*14]
        
        # Reshape for the delta block
        LRR = fcorrs.shape[3]
        fcorrs_ = fcorrs.permute(0, 2, 1, 3).reshape(B * N, S, LRR)  # [B*N, S, LRR]
        flows_ = (coords[:, 1:] - coords[:, :-1]).permute(0, 2, 1, 3).reshape(B * N, S - 1, 2)
        flows_ = torch.cat([flows_, flows_[:, -1:]], dim=1)  # [B*N, S, 2]

        # print(fcorrs_.shape, flows_.shape, frame_flow.shape, xys0.shape)
        # raise KeyboardInterrupt
        # Compute delta coordinates
        delta_coords_ = self.delta_block_spatial(fcorrs_, flows_, frame_flow, xys0)  # [B*N, S, 2]
        delta_coords_ = delta_coords_.reshape(B, N, S, 2).permute(0, 2, 1, 3)
        coords = coords + delta_coords_
        
        # Scale back to original resolution
        #print(points_0.shape, points_0.sum(), coords.shape, q_frame)
        trajs_e = coords * float(self.stride)  # [B, S, N, 2]
        trajs_e[:,q_frame] = points_0
        return trajs_e

        
    def forward(
        self, 
        rgbs:torch.Tensor, 
        points_0:torch.Tensor, 
        iters:int=4, 
        beautify:bool=False,
        ):
        """ Forward pass of the model

        Args:
            rgbs (tensor): a sequence of frames of shape [B, S, C, H, W]. S must be divisible by 8. range: [0 - 255]
            points_0 (tensor): a list of points on the first frame of the given video [B, N, 2]. range: [0 - H/W]
            
        Returns:
            _type_: _description_
        """  
        B,N,D = points_0.shape
        if D==3:
            q_frame:int = int(points_0[0,0,0].long().item())
            points_0 = points_0[..., 1:]
        elif D==2:
            q_frame:int=0
        else:
            q_frame:int=-1
            assert 1, "Dimension should be 2 or 3"

        B, S, C, H, W = rgbs.shape
        assert (C==1), "Number of channel should be 1 for echo frames"
        #normalizing in [-1.0, 1.0]
        rgbs = 2 * (rgbs / 255.0) - 1.0 #normalization within -1.0 to 1.0

        coords_init = self.initialize_(rgbs, points_0, q_frame=q_frame)
        
        coord_predictions1 = [] # for loss
        coord_predictions1.append(coords_init)
        
        #creating the fine feature maps for all the frames
        H_ = H//int(self.stride/self.scale)
        W_ = W//int(self.stride/self.scale)
        rgbs_ = rgbs.reshape(B*S, C, H, W)
        if self.feature_extractor_chunk_size > 0:
            fmaps_list = []
            chunk_size = self.feature_extractor_chunk_size
            for start_idx in range(0, rgbs_.shape[0], chunk_size):
                rgb_chunk = rgbs_[start_idx:start_idx + chunk_size]
                fmaps_list.append(self.fnet_fine(rgb_chunk))
            fmaps_ = torch.cat(fmaps_list, dim=0)

        else:
            fmaps_ = self.fnet_fine(rgbs_)
        fmaps = fmaps_.reshape(B, S, self.latent_dim, H_, W_) # the fine fmaps
        
        rgbs_ = rgbs_.reshape(B, S, C, H, W)


        coords = coords_init.clone()/(self.stride/self.scale) # B,S,N,2

        # #computing cost volume at multi-scale using pyramids
        fcorr_fn1 = CorrBlock(fmaps, num_levels=self.corr_levels, radius=self.corr_radius)
        fcorr_fn2 = CorrBlock(fmaps, num_levels=self.corr_levels, radius=self.corr_radius)
        fcorr_fn4 = CorrBlock(fmaps, num_levels=self.corr_levels, radius=self.corr_radius)

        # #Features for all the points from the first frame
        f_vecs0 = bilinear_sample2d(fmaps[:,q_frame],  coords[:,q_frame,:,0], coords[:,q_frame,:,1]).permute(0, 2, 1) # B,N,C  feature vectors in query frame
        feats1 = f_vecs0.unsqueeze(1).repeat(1, S, 1, 1) # B,S,N,C
        
        fcorr_fn1.corr(feats1, chunk_size=self.feature_extractor_chunk_size) #computing cross-correlations

        coords_bak = coords.clone()
        

        rgbs_ = rgbs_.squeeze(2)
        frame_flow = torch.cat([rgbs_[:,-1:], (rgbs_[:, 1:] - rgbs_[:, :-1])], dim=1)#.reshape(B*S, H, W)#.unsqueeze(1)
        frame_flow = F.interpolate(frame_flow, (50, 60), mode="bilinear").reshape(B, 1, S, 50 * 60).repeat(1, N, 1, 1).reshape(B*N, S, 50 * 60)
        xys0 = coords_bak[:, q_frame].unsqueeze(2).repeat(1, 1, S, 1).reshape(B*N, S, 2)


        for itr in range(iters):
            coords = coords.detach()

            # timestep indices
            inds2 = (torch.arange(S)-2).clip(min=0)
            inds4 = (torch.arange(S)-4).clip(min=0)
            # coordinates at these timesteps
            coords2_ = coords[:,inds2].reshape(B*S,N,2)
            coords4_ = coords[:,inds4].reshape(B*S,N,2)
            # featuremaps at these timesteps
            fmaps2_ = fmaps[:,inds2].reshape(B*S,self.latent_dim,H_,W_)
            fmaps4_ = fmaps[:,inds4].reshape(B*S,self.latent_dim,H_,W_)
            # features at these coords/times
            feats2_ = bilinear_sample2d(fmaps2_, coords2_[:,:,0], coords2_[:,:,1]).permute(0, 2, 1) # B*S, N, C
            feats2 = feats2_.reshape(B,S,N,self.latent_dim)
            feats4_ = bilinear_sample2d(fmaps4_, coords4_[:,:,0], coords4_[:,:,1]).permute(0, 2, 1) # B*S, N, C
            feats4 = feats4_.reshape(B,S,N,self.latent_dim)

            fcorr_fn2.corr(feats2, chunk_size=self.feature_extractor_chunk_size)
            fcorr_fn4.corr(feats4, chunk_size=self.feature_extractor_chunk_size)

            # now we want costs at the current locations
            fcorrs1 = fcorr_fn1.sample(coords, chunk_size=self.feature_extractor_chunk_size) # B,S,N,LRR
            fcorrs2 = fcorr_fn2.sample(coords, chunk_size=self.feature_extractor_chunk_size) # B,S,N,LRR
            fcorrs4 = fcorr_fn4.sample(coords, chunk_size=self.feature_extractor_chunk_size) # B,S,N,LRR
            LRR = fcorrs1.shape[3]

            # we want everything in the format B*N, S, C
            fcorrs1_ = fcorrs1.permute(0, 2, 1, 3).reshape(B*N, S, LRR)
            fcorrs2_ = fcorrs2.permute(0, 2, 1, 3).reshape(B*N, S, LRR)
            fcorrs4_ = fcorrs4.permute(0, 2, 1, 3).reshape(B*N, S, LRR)
            fcorrs_ = torch.cat([fcorrs1_, fcorrs2_, fcorrs4_], dim=2)
            fcorrs_ = self.linear(fcorrs_)
            flows_ = (coords[:,1:] - coords[:,:-1]).permute(0,2,1,3).reshape(B*N, S-1, 2)
            flows_ = torch.cat([flows_, flows_[:,-1:]], dim=1) # B*N,S,2

            delta_coords_ = self.delta_block_temporal(fcorrs_, flows_, frame_flow, xys0) # B*N,S,2

            if beautify and itr > 3*iters//4:
                # this smooths the results a bit, but does not really help perf
                delta_coords_ = delta_coords_ * 0.5
            
            coords = coords + delta_coords_.reshape(B, N, S, 2).permute(0,2,1,3)
            coords_iter = coords * (self.stride/self.scale)
            #print(coords_iter.shape, points_0.shape, q_frame)
            #coords_iter[:,q_frame] = points_0
            #raise KeyboardInterrupt
            coord_predictions1.append(coords_iter)
        return coord_predictions1