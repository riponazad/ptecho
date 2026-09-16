""" 
Created on Friday Apr 26 2025
@author: Azad Md Abulkalam
@location: ISB, NTNU

SpeckNet architecture.
"""
import torch
from torch import nn
import functools
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from models.blocks import (
    FeatureGrids, QueryFeatures, ResNet, ExtraConvs
)
from utils.utils_ import (
    convert_grid_coordinates, map_coordinates_3d, is_same_res, bilinear,
    heatmaps_to_points,
)



class SpeckNet(nn.Module):
    """SpeckNet Model"""
    def __init__(
        self,
        initial_resolution: Tuple[int, int] = (512, 512),
        blocks_per_group: Sequence[int] = (2, 2, 2, 2),
        feature_extractor_chunk_size: int = 64,
        softmax_temperature: float = 20.0,
        multi_scale_level: int = 2, # options:{0,1,2,3} 
    ):
        super(SpeckNet, self).__init__()

        self.initial_resolution = tuple(initial_resolution)
        self.feature_extractor_chunk_size = feature_extractor_chunk_size
        self.softmax_temperature = softmax_temperature
        self.multi_scale_level = multi_scale_level

        highres_dim = 64
        lowres_dim = 64
        strides = (1, 2, 1, 2)
        channels_per_group = (32, highres_dim, 64, lowres_dim)
        use_projection = (True, True, True, True)

        self.resnet_torch = ResNet(
            blocks_per_group=blocks_per_group,
            channels_per_group=channels_per_group,
            use_projection=use_projection,
            strides=strides,
        )
        self.extra_convs = ExtraConvs(channel_dim=lowres_dim)
        
    def forward(
        self,
        video: torch.Tensor,
        query_points: torch.Tensor = None,
        query_chunk_size: Optional[int] = 64,
    ) -> Mapping[str, torch.Tensor]:
        """Runs the forward pass of the model.

        Args:
            video (torch.Tensor): 
            query_points (torch.Tensor): _description_
            trajs_g (torch.Tensor, optional): _description_. Defaults to None.
            vis_g (torch.Tensor, optional): _description_. Defaults to None.

        Returns:
            Mapping[str, torch.Tensor]: _description_
        """
        video = (2 * (video / 255.0) - 1.0) #[-1, 1]
        feature_grids = self.get_feature_grids(video)

        query_features = self.get_query_features(
            video,
            query_points,
            feature_grids,
        )

        trajectories = self.estimate_trajectories(
            video.shape[-3:-1],
            feature_grids,
            query_features,
            query_points,
            query_chunk_size,
        )
        
        out = dict(
            occlusion=trajectories['occlusion'],
            tracks=trajectories['tracks'],
            expected_dist=trajectories['expected_dist'],
        )

        return out

    def get_query_features(
        self,
        video: torch.Tensor,
        query_points: torch.Tensor,
        feature_grids: Optional[FeatureGrids] = None,
    ) -> QueryFeatures:
        """Computes query features, which can be used for estimate_trajectories.

        Args:
        video: A 5-D tensor representing a batch of sequences of images.
        is_training: Whether we are training.
        query_points: The query points for which we compute tracks.
        feature_grids: If passed, we'll use these feature grids rather than
            computing new ones.
        refinement_resolutions: A list of (height, width) tuples.  Refinement will
            be repeated at each specified resolution, in order to achieve high
            accuracy on resolutions higher than what TAPIR was trained on. If None,
            reasonable refinement resolutions will be inferred from the input video
            size.

        Returns:
        A QueryFeatures object which contains the required features for every
            required resolution.
        """
        query_feats = []
        for i in range(self.multi_scale_level):
            position_in_grid = convert_grid_coordinates(
                query_points,
                video.shape[1:4],
                feature_grids.feats[i].shape[1:4],
                coordinate_format='tyx',
            )
            interp_features = map_coordinates_3d(
                feature_grids.feats[i], position_in_grid
            )
            query_feats.append(interp_features)

        return QueryFeatures(feats=query_feats, lowres=None, hires=None, resolutions=None)

    
    def get_feature_grids(self, video: torch.Tensor) -> FeatureGrids:
        """Computes feature grids.
        Args:
        video: A 5-D tensor representing a batch of sequences of images.
        refinement_resolutions: A list of (height, width) tuples. Refinement will
            be repeated at each specified resolution, to achieve high accuracy on
            resolutions higher than what TAPIR was trained on. If None, reasonable
            refinement resolutions will be inferred from the input video size.

        Returns:if __main__e one more feature grid
        than there are refinement_resolutions, because there is always a
        feature grid computed for TAP-Net initialization.
        """
        # if refinement_resolutions is None:
        #     refinement_resolutions = utls.generate_default_resolutions(
        #         video.shape[2:4], self.initial_resolution
        #     )
    
        all_required_resolutions = [self.initial_resolution]
        
        
        curr_resolution = (-1, -1)
        feats = []
        for _ in range(self.multi_scale_level):
            feats.append([])

        video_resize = None
        for resolution in all_required_resolutions:
            if resolution[0] % 8 != 0 or resolution[1] % 8 != 0:
                raise ValueError('Image resolution must be a multiple of 8.')

            if not is_same_res(curr_resolution, resolution):
                if is_same_res(curr_resolution, video.shape[-3:-1]):
                    video_resize = video
                else:
                    video_resize = bilinear(video, resolution)

                curr_resolution = resolution
                n, f, h, w, c = video_resize.shape
                # out = self.resnet3d(video_resize.permute(0, 4, 1, 2, 3))
                # for i in range(self.multi_scale_level):
                #     feats[i] = out[f'resnet_unit_{3-i}'].permute(0, 2, 3, 4, 1)
                # latent = out['resnet_unit_3'].permute(0, 2, 3, 4, 1)
                # print(video_resize.shape, hh['resnet_unit_3'].shape)
                # raise KeyboardInterrupt
                
                video_resize = video_resize.view(n*f, h, w, c).permute(0, 3, 1, 2)
                if self.feature_extractor_chunk_size > 0:
                    feats_list = []
                    feats_list = [[] for _ in range(self.multi_scale_level)]
                    chunk_size = self.feature_extractor_chunk_size
                    for start_idx in range(0, video_resize.shape[0], chunk_size):
                        video_chunk = video_resize[start_idx:start_idx + chunk_size]
                        resnet_out = self.resnet_torch(video_chunk)
                        for i in range(self.multi_scale_level):
                            feats_list[i].append(resnet_out[f'resnet_unit_{3-i}'].permute(0, 2, 3, 1))#.detach()
                        # print(u2.shape, u3.shape)
                        # raise KeyboardInterrupt
                    for i in range(self.multi_scale_level):
                        feats[i] = torch.cat(feats_list[i], dim=0)
                    

                else:
                    resnet_out = self.resnet_torch(video_resize)
                    for i in range(self.multi_scale_level):
                        feats[i] = resnet_out[f'resnet_unit_{3-i}'].permute(0, 2, 3, 1)#.detach()
                    
                # print(latent.shape)
                # raise KeyboardInterrupt
                for i in range(self.multi_scale_level):
                    feats[i] = self.extra_convs(feats[i])
                    feats[i] = feats[i] / torch.sqrt(
                        torch.maximum(
                            torch.sum(torch.square(feats[i]), axis=-1, keepdims=True),
                            torch.tensor(1e-12, device=feats[i].device),
                        )
                    )
            
            # Reshape latent features back to batch format and add to feature grid
            for i in range(self.multi_scale_level):
                feats[i] = feats[i].view(n, f, *feats[i].shape[1:])
                
        return FeatureGrids(feats=feats, lowres=None, hires=None, resolutions=None)
        

    def estimate_trajectories(
        self,
        video_size: Tuple[int, int],
        feature_grids: FeatureGrids,
        query_features: QueryFeatures,
        query_points_in_video: Optional[torch.Tensor],
        query_chunk_size: Optional[int] = None,
    ) -> Mapping[str, Any]:
        """Estimates trajectories given features for a video and query features.

        Args:
        video_size: A 2-tuple containing the original [height, width] of the
            video.  Predictions will be scaled with respect to this resolution.
        is_training: Whether we are training.
        feature_grids: a FeatureGrids object computed for the given video.
        query_features: a QueryFeatures object computed for the query points.
        query_points_in_video: If provided, assume that the query points come from
            the same video as feature_grids, and therefore constrain the resulting
            trajectories to (approximately) pass through them.
        query_chunk_size: When computing cost volumes, break the queries into
            chunks of this size to save memory.

        Returns:
        A dict of outputs, including:
            occlusion: Occlusion logits, of shape [batch, num_queries, num_frames]
            where higher indicates more likely to be occluded.
            tracks: predicted point locations, of shape
            [batch, num_queries, num_frames, 2], where each point is [x, y]
            in raster coordinates
            expected_dist: uncertainty estimate logits, of shape
            [batch, num_queries, num_frames], where higher indicates more likely
            to be far from the correct answer.
        """

        def train2orig(x):
            return convert_grid_coordinates(
                x,
                self.initial_resolution[::-1],
                video_size[::-1],
                coordinate_format='xy',
            )
        infer = functools.partial(
            self.tracks_from_cost_volume,
            #im_shp=feature_grids.lowres[0].shape[0:2]
            im_shp=feature_grids.feats[-1].shape[0:2]
            + self.initial_resolution
            + (3,),
        )
        trajs_e = []
        feat_grids = []
        q_feats = []
        for i in range(self.multi_scale_level):
            feat_grids.append(feature_grids.feats[i])
            q_feats.append(query_features.feats[i])      
        

        num_queries = q_feats[0].shape[1]
        perm = torch.randperm(num_queries)
        inv_perm = torch.zeros_like(perm)
        inv_perm[perm] = torch.arange(num_queries)

        chunk_list = []
        for ch in range(0, num_queries, query_chunk_size):
            perm_chunk = perm[ch : ch + query_chunk_size]
            # chunk = q_feats[0][:, perm_chunk]
            chunk_list = [q_feats[i][:, perm_chunk] for i in range(self.multi_scale_level)]
            # chunk_list.append(q_feats[0][0][:, perm_chunk])
            # chunk_list.append(q_feats[1][0][:, perm_chunk])
            
            if query_points_in_video is not None:
                infer_query_points = query_points_in_video[
                    :, perm[ch : ch + query_chunk_size]
                ]
                num_frames = feat_grids[0][0].shape[1]
                infer_query_points = convert_grid_coordinates(
                    infer_query_points,
                    (num_frames,) + video_size,
                    (num_frames,) + self.initial_resolution,
                    coordinate_format='tyx',
                )
            else:
                infer_query_points = None
            

            # print(feat_grids[0].shape[0:2]
            # + self.initial_resolution
            # + (3,), infer_query_points.shape)
            # raise KeyboardInterrupt
            # points, occlusion, expected_dist = infer(
            #     chunk,
            #     feat_grids[0],
            #     infer_query_points,
            # )
            points = infer(
                chunk_list,
                feat_grids,
                infer_query_points,
            )
            trajs_e.append(train2orig(points))
            chunk_list = []
        
        trajs_e = torch.cat(trajs_e, dim=1)[:, inv_perm]
        
        out = dict(
            occlusion=None,
            tracks=trajs_e,
            expected_dist=None,
        )
        return out


    def tracks_from_cost_volume(
        self,
        interp_feature: List[torch.Tensor],
        feature_grid: List[torch.Tensor],
        query_points: Optional[torch.Tensor],
        im_shp=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Converts features into tracks by computing a cost volume.

        The computed cost volume will have shape
        [batch, num_queries, time, height, width], which can be very
        memory intensive.

        Args:
        interp_feature: A tensor of features for each query point, of shape
            [batch, num_queries, channels, heads].
        feature_grid: A tensor of features for the video, of shape [batch, time,
            height, width, channels, heads].
        query_points: When computing tracks, we assume these points are given as
            ground truth and we reproduce them exactly.  This is a set of points of
            shape [batch, num_points, 3], where each entry is [t, y, x] in frame/
            raster coordinates.
        im_shp: The shape of the original image, i.e., [batch, num_frames, time,
            height, width, 3].

        Returns:
        A 2-tuple of the inferred points (of shape
            [batch, num_points, num_frames, 2] where each point is [x, y]) and
            inferred occlusion (of shape [batch, num_points, num_frames], where
            each is a logit where higher means occluded)
        """
        B, T, H, W, _ = feature_grid[-1].shape
        cost_volume_pyramids = []
        for i in range(self.multi_scale_level):
            _, _, H_, W_, _ = feature_grid[i].shape
            cost_volume = torch.einsum(
                'bnc,bthwc->tbnhw',
                interp_feature[i],
                feature_grid[i],
            ) ** 4.0
            # cost_volume = (cost_volume + 1) / 2  # Now in [0, 1]
            # cost_volume **= 15.0
            # frame_list = torch.linspace(0, T - 1, steps=7).long()
            # visualize_feature_maps(
            #     generate_attn_maps(cost_volume.permute(1,2,0,3,4), resize=None),
            #     #cost_volume.permute(1,2,0,3,4)[:,20],
            #     b=0, 
            #     c_list=frame_list, 
            #     cmap='gray',
            #     save_path=f'cost_volumes_{i}',
            # )
            #print(f"Cost Volume: {cost_volume.shape}, {cost_volume.min()}, {cost_volume.max()}")
            cost_volume = bilinear(cost_volume.permute(0,1, 3, 4, 2), (H, W)).permute(0,1, 4, 2, 3)
            cost_volume_pyramids.append(cost_volume)
        
        for i in range(len(cost_volume_pyramids)-1):
            cost_volume *= cost_volume_pyramids[i]
        cost_volume = cost_volume**2.0
        # visualize_feature_maps(
        #     generate_attn_maps(cost_volume.permute(1,2,0,3,4), resize=None),
        #     #cost_volume.permute(1,2,0,3,4)[:,20],
        #     b=0, 
        #     c_list=frame_list, 
        #     cmap='gray',
        #     save_path=f'cost_volumes',
        # )
        #print(f"Cost Volume: {cost_volume.shape}")
        
        points = heatmaps_to_points(cost_volume.permute(1,2,0,3,4), im_shp, query_points=query_points, threshold=5)
        
        return points
