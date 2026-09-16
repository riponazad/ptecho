"""EchoTracker2 models definition."""

import functools
from typing import Any, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union, Dict

import torch
from torch import nn
import torch.nn.functional as F


from models.blocks import CMDTop, ResNet_echo2, KNPTransformer

from utils.utils_ import (
    convert_grid_coordinates, map_coordinates_3d, is_same_res, bilinear, 
    generate_default_resolutions, map_coordinates_2d
)


@torch.jit.script
def posenc(x: torch.Tensor, min_deg: int, max_deg: int, legacy_posenc_order: bool = False) -> torch.Tensor:
    """Positional encoding compatible with TorchScript."""
    if min_deg == max_deg:
        return x

    # Generate scales
    scales = 2.0 ** torch.arange(min_deg, max_deg, dtype=x.dtype, device=x.device)  # [num_scales]

    if legacy_posenc_order:
        # [ ..., 1, dim] * [num_scales, 1] -> [ ..., num_scales, dim]
        xb = x.unsqueeze(-2) * scales.view(-1, 1)
        xb_stack = torch.stack([xb, xb + 0.5 * torch.pi], dim=-2)  # [..., 2, num_scales, dim]
        four_feat = xb_stack.flatten(start_dim=-2)  # flatten last 2 dims
    else:
        xb = (x.unsqueeze(-2) * scales.view(-1, 1)).flatten(-2)  # [..., num_scales * dim]
        four_feat = torch.sin(torch.cat([xb, xb + 0.5 * torch.pi], dim=-1))

    return torch.cat([x, four_feat], dim=-1)


class FeatureGrids(NamedTuple):
  """Feature grids for a video, used to compute trajectories.

  These are per-frame outputs of the encoding resnet.

  Attributes:
    lowres: Low-resolution features, one for each resolution; 256 channels.
    hires: High-resolution features, one for each resolution; 64 channels.
    resolutions: Resolutions used for trajectory computation.  There will be one
      entry for the initialization, and then an entry for each PIPs refinement
      resolution.
  """

  # lowres: Sequence[torch.Tensor]
  # hires: Sequence[torch.Tensor]
  # highest: Sequence[torch.Tensor]
  # resolutions: Sequence[Tuple[int, int]]
  lowres: List[torch.Tensor]
  hires: List[torch.Tensor]
  highest: List[torch.Tensor]
  resolutions: List[Tuple[int, int]]


class QueryFeatures(NamedTuple):
  """Query features used to compute trajectories.

  These are sampled from the query frames and are a full descriptor of the
  tracked points. They can be acquired from a query image and then reused in a
  separate video.

  Attributes:
    lowres: Low-resolution features, one for each resolution; each has shape
      [batch, num_query_points, 256]
    hires: High-resolution features, one for each resolution; each has shape
      [batch, num_query_points, 64]
    resolutions: Resolutions used for trajectory computation.  There will be one
      entry for the initialization, and then an entry for each PIPs refinement
      resolution.
  """

  # lowres: Sequence[torch.Tensor]
  # hires: Sequence[torch.Tensor]
  # highest: Sequence[torch.Tensor]
  # lowres_supp: Sequence[torch.Tensor]
  # hires_supp: Sequence[torch.Tensor]
  # highest_supp: Sequence[torch.Tensor]
  # resolutions: Sequence[Tuple[int, int]]
  # lowres: List[torch.Tensor]
  # hires: List[torch.Tensor]
  # highest: List[torch.Tensor]
  lowres_supp: List[torch.Tensor]
  hires_supp: List[torch.Tensor]
  highest_supp: List[torch.Tensor]
  resolutions: List[Tuple[int, int]]


class EchoTracker2(nn.Module):
  """EchoTracker2 model."""

  def __init__(
      self,
      num_pips_iter: int = 4,
      pyramid_level: int = 0,
      patch_size: int = 7,
      initial_resolution: Tuple[int, int] = (256, 256),
      feature_extractor_chunk_size: int = 256,
      sp_attn:bool = False,
      itsm_resnet:bool = False,
  ):
    super().__init__()
    model_params = {
      'dim': 384,
      'num_heads': 6,
      'num_layers': 3,
    }
    cmdtop_params = {
      'in_channel': patch_size*patch_size,#49,
      'out_channels': (64, 128, 128),
      'kernel_shapes': (3, 3, 2), 
      'strides': (2, 2, 2),
    }

    self.highres_dim = 128
    self.lowres_dim = 256
    
    
    self.num_pips_iter = num_pips_iter
    self.pyramid_level = pyramid_level
    self.patch_size = patch_size
    self.initial_resolution = tuple(initial_resolution)
    self.feature_extractor_chunk_size = feature_extractor_chunk_size
    self.itsm_resnet = itsm_resnet
    
    highres_dim = 128
    lowres_dim = 256
    strides = (1, 2, 2, 1)
    blocks_per_group = (2, 2, 2, 2)
    channels_per_group = (64, highres_dim, 256, lowres_dim)
    use_projection = (True, True, True, True)

    self.resnet_torch = ResNet_echo2(
        blocks_per_group=blocks_per_group,
        channels_per_group=channels_per_group,
        use_projection=use_projection,
        strides=strides,
        temp_adapter=itsm_resnet,
    )
    
    dim_input = 256*3 + 84
    
    self.torch_pips_mixer = KNPTransformer(
      input_channels=dim_input,
      output_channels=2 + self.highres_dim + self.lowres_dim,
      sp_attn=sp_attn,
      **model_params
    )
    
    self.cmdtop = nn.ModuleList([
      CMDTop(
        **cmdtop_params
      ) for _ in range(3)
    ])

  def forward(
      self,
      video: torch.Tensor,
      query_points: torch.Tensor,
      is_training: bool = False,
      query_chunk_size: int = 64,
      #refinement_resolutions: Optional[List[Tuple[int, int]]] = None,
  ) -> Dict[str, torch.Tensor]:
    """Runs a forward pass of the model.

    Args:
      video: A 5-D tensor representing a batch of sequences of images.
      query_points: The query points for which we compute tracks.
      is_training: Whether we are training.
      query_chunk_size: When computing cost volumes, break the queries into
        chunks of this size to save memory.
      get_query_feats: Return query features for other losses like contrastive.
        Not supported in the current version.
      refinement_resolutions: A list of (height, width) tuples.  Refinement will
        be repeated at each specified resolution, in order to achieve high
        accuracy on resolutions higher than what TAPIR was trained on. If None,
        reasonable refinement resolutions will be inferred from the input video
        size.

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
    video = (2 * (video / 255.0) - 1.0)
    #query_points = query_points[..., [0, 2, 1]] #(t, x, y) -> (t, y, x)

    feature_grids = self.get_feature_grids(
        video,
        is_training,
        #refinement_resolutions,
    )
    # for i in range(len(feature_grids.highest)):
    #   print(f"{i}: {feature_grids.lowres[i].shape}, {feature_grids.hires[i].shape},{feature_grids.highest[i].shape}")
    # raise KeyboardInterrupt
    query_features = self.get_query_features(
        video,
        is_training,
        query_points,
        feature_grids,
        #refinement_resolutions,
    )
    # for i in range(len(query_features.highest_supp)):
    #   print(f"{i}: {query_features.lowres_supp[i].shape}, {query_features.hires_supp[i].shape},{query_features.highest_supp[i].shape}")
    # raise KeyboardInterrupt
    #print(video.shape)
    
       

    trajectories = self.estimate_trajectories(
        (video.shape[-3],video.shape[-2]),
        is_training,
        feature_grids,
        query_features,
        query_points,
        query_chunk_size,
    )
    p = self.num_pips_iter
    out = dict(
        tracks=torch.mean(torch.stack(trajectories['tracks'][p-1::p]), dim=0),
        # unrefined_occlusion=trajectories['occlusion'][:-1],
        # unrefined_tracks=trajectories['tracks'][:-1],
        # unrefined_expected_dist=trajectories['expected_dist'][:-1],
    )
    #query_points = query_points[..., [0, 2, 1]] #(t, y, x) -> (t, x, y)
    return out

  def get_query_features(
      self,
      video: torch.Tensor,
      is_training: bool,
      query_points: torch.Tensor,
      feature_grids: Optional[FeatureGrids] = None,
      #refinement_resolutions: Optional[List[Tuple[int, int]]] = None,
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

    if feature_grids is None:
      feature_grids = self.get_feature_grids(
          video,
          is_training=is_training,
          #refinement_resolutions=refinement_resolutions,
      )

    feature_grid = feature_grids.lowres
    hires_feats = feature_grids.hires
    highest_feats = feature_grids.highest
    resize_im_shape = feature_grids.resolutions

    shape = video.shape
    # shape is [batch_size, time, height, width, channels]; conversion needs
    # [time, width, height]
    curr_resolution = (-1, -1)
    query_supp: List[torch.Tensor] = torch.jit.annotate(List[torch.Tensor], [])
    hires_query_supp: List[torch.Tensor] = torch.jit.annotate(List[torch.Tensor], [])
    highest_query_supp: List[torch.Tensor] = torch.jit.annotate(List[torch.Tensor], [])
    for i, resolution in enumerate(resize_im_shape):
      if is_same_res(curr_resolution, resolution):
        query_supp.append(query_supp[-1])
        hires_query_supp.append(hires_query_supp[-1])
        highest_query_supp.append(highest_query_supp[-1])
        continue
      position_in_grid = convert_grid_coordinates(
          query_points,
          shape[1:4],
          feature_grid[i].shape[1:4],
          coordinate_format='tyx',
      )
      position_in_grid_hires = convert_grid_coordinates(
          query_points,
          shape[1:4],
          hires_feats[i].shape[1:4],
          coordinate_format='tyx',
      )
      position_in_grid_highest = convert_grid_coordinates(
          query_points,
          shape[1:4],
          highest_feats[i].shape[1:4],
          coordinate_format='tyx',
      )

      support_size = self.patch_size #7
      ctxx, ctxy = torch.meshgrid(
        torch.arange(-(support_size // 2), support_size // 2 + 1), 
        torch.arange(-(support_size // 2), support_size // 2 + 1),
        indexing='xy',
      )
      ctx = torch.stack([torch.zeros_like(ctxy), ctxy, ctxx], dim=-1)
      ctx = torch.reshape(ctx, [-1, 3]).to(video.device) # s*s 3

      position_support = position_in_grid[..., None, :] + ctx[None, None, ...] # b n s*s 3
      # position_support = rearrange(position_support, 'b n s c -> b (n s) c')
      b, n, s, c = position_support.shape
      position_support = position_support.reshape(b, n*s,c)
      interp_supp = map_coordinates_3d(
          feature_grid[i], position_support
      )
      #interp_supp = rearrange(interp_supp, 'b (n h w) c -> b n h w c', h=support_size, w=support_size)
      b, nhw, c = interp_supp.shape
      n = nhw // (support_size * support_size)
      interp_supp = interp_supp.reshape(b, n, support_size, support_size, c)

      position_support_hires = position_in_grid_hires[..., None, :] + ctx[None, None, ...]
      #position_support_hires = rearrange(position_support_hires, 'b n s c -> b (n s) c')
      b, n, s, c = position_support_hires.shape
      position_support_hires = position_support_hires.reshape(b, n*s,c)
      hires_interp_supp = map_coordinates_3d(
          hires_feats[i], position_support_hires
      )
      #hires_interp_supp = rearrange(hires_interp_supp, 'b (n h w) c -> b n h w c', h=support_size, w=support_size)
      b, nhw, c = hires_interp_supp.shape
      n = nhw // (support_size * support_size)
      hires_interp_supp = hires_interp_supp.reshape(b, n, support_size, support_size, c)

      position_support_highest = position_in_grid_highest[..., None, :] + ctx[None, None, ...]
      #position_support_highest = rearrange(position_support_highest, 'b n s c -> b (n s) c')
      b, n, s, c = position_support_highest.shape
      position_support_highest = position_support_highest.reshape(b, n*s,c)
      highest_interp_supp = map_coordinates_3d(
          highest_feats[i], position_support_highest
      )
      #highest_interp_supp = rearrange(highest_interp_supp, 'b (n h w) c -> b n h w c', h=support_size, w=support_size)
      b, nhw, c = highest_interp_supp.shape
      n = nhw // (support_size * support_size)
      highest_interp_supp = highest_interp_supp.reshape(b, n, support_size, support_size, c)

      query_supp.append(interp_supp)
      hires_query_supp.append(hires_interp_supp)
      highest_query_supp.append(highest_interp_supp)

    return QueryFeatures(
        lowres_supp=query_supp, hires_supp=hires_query_supp, highest_supp=highest_query_supp, resolutions=resize_im_shape,
    )

  def get_feature_grids(
      self,
      video: torch.Tensor,
      is_training: Optional[bool] = False,
      #refinement_resolutions: Optional[List[Tuple[int, int]]] = None,
  ) -> FeatureGrids:
    """Computes feature grids.

    Args:
      video: A 5-D tensor representing a batch of sequences of images.
      is_training: Whether we are training.
      refinement_resolutions: A list of (height, width) tuples. Refinement will
        be repeated at each specified resolution, to achieve high accuracy on
        resolutions higher than what TAPIR was trained on. If None, reasonable
        refinement resolutions will be inferred from the input video size.

    Returns:
      A FeatureGrids object containing the required features for every
      required resolution. Note that there will be one more feature grid
      than there are refinement_resolutions, because there is always a
      feature grid computed for TAP-Net initialization.
    """
    del is_training
    #if refinement_resolutions is None:
    # refinement_resolutions = utils.generate_default_resolutions(
    #     video.shape[2:4], self.initial_resolution
    # )
    refinement_resolutions = generate_default_resolutions(
        (video.shape[2],video.shape[3]), self.initial_resolution
    )

    all_required_resolutions: List[Tuple[int, int]] = []
    all_required_resolutions.extend(refinement_resolutions)

    feature_grid:List[torch.Tensor] = torch.jit.annotate(List[torch.Tensor], [])
    hires_feats: List[torch.Tensor] = torch.jit.annotate(List[torch.Tensor], [])
    highest_feats: List[torch.Tensor] = torch.jit.annotate(List[torch.Tensor], [])
    resize_im_shape: List[Tuple[int, int]] = torch.jit.annotate(List[Tuple[int, int]], [])
    curr_resolution = (-1, -1)

    #latent = None
    latent: torch.Tensor = torch.jit.annotate(torch.Tensor, torch.empty(0))
    #hires = None
    hires: torch.Tensor = torch.jit.annotate(torch.Tensor, torch.empty(0))
    #highest = None
    highest: torch.Tensor = torch.jit.annotate(torch.Tensor, torch.empty(0))
    #video_resize = None
    video_resize: torch.Tensor = torch.jit.annotate(torch.Tensor, torch.empty(0))
    for resolution in all_required_resolutions:
      if resolution[0] % 8 != 0 or resolution[1] % 8 != 0:
        raise ValueError('Image resolution must be a multiple of 8.')

      if not is_same_res(curr_resolution, resolution):
        # if utils.is_same_res(curr_resolution, video.shape[-3:-1]):
        if is_same_res(curr_resolution, (video.shape[-3], video.shape[-2])):
          video_resize = video
        else:
          video_resize = bilinear(video, resolution)

        curr_resolution = resolution
        n, f, h, w, c = video_resize.shape
        #video_resize = video_resize.permute(0, 1, 4, 2, 3)
        video_resize = video_resize.reshape(n*f, h, w, c).permute(0, 3, 1, 2)
        

        if self.feature_extractor_chunk_size > 0:
          latent_list = []
          hires_list = []
          highest_list = []
          chunk_size = self.feature_extractor_chunk_size
          for start_idx in range(0, video_resize.shape[0], chunk_size):
          #for start_idx in range(0, video_resize.shape[1], chunk_size):
            video_chunk = video_resize[start_idx:start_idx + chunk_size]
            #video_chunk = video_resize[:,start_idx:start_idx + chunk_size]
            resnet_out = self.resnet_torch(video_chunk)
            u3 = resnet_out['resnet_unit_3'].permute(0, 2, 3, 1)
            latent_list.append(u3)
            u1 = resnet_out['resnet_unit_1'].permute(0, 2, 3, 1)
            hires_list.append(u1)
            u0 = resnet_out['resnet_unit_0'].permute(0, 2, 3, 1)
            highest_list.append(u0)

          latent = torch.cat(latent_list, dim=0)
          hires = torch.cat(hires_list, dim=0)
          highest = torch.cat(highest_list, dim=0)

        else:
          resnet_out = self.resnet_torch(video_resize)
          latent = resnet_out['resnet_unit_3'].permute(0, 2, 3, 1)
          hires = resnet_out['resnet_unit_1'].permute(0, 2, 3, 1)
          highest = resnet_out['resnet_unit_0'].permute(0, 2, 3, 1)

        latent = latent / torch.sqrt(
            torch.maximum(
                torch.sum(torch.square(latent), dim=-1, keepdim=True),
                torch.tensor(1e-12, device=latent.device),
            )
        )
        hires = hires / torch.sqrt(
            torch.maximum(
                torch.sum(torch.square(hires), dim=-1, keepdim=True),
                torch.tensor(1e-12, device=hires.device),
            )
        )
        highest = highest / torch.sqrt(
            torch.maximum(
                torch.sum(torch.square(highest), dim=-1, keepdim=True),
                torch.tensor(1e-12, device=highest.device),
            )
        )

        # latent = latent.view(n, f, *latent.shape[1:])
        # hires = hires.view(n, f, *hires.shape[1:])
        # highest = highest.view(n, f, *highest.shape[1:])
        latent = latent.view(n, f, latent.shape[1], latent.shape[2], latent.shape[3])
        hires = hires.view(n, f, hires.shape[1], hires.shape[2], hires.shape[3])
        highest = highest.view(n, f, highest.shape[1], highest.shape[2], highest.shape[3])

      feature_grid.append(latent)
      hires_feats.append(hires)
      highest_feats.append(highest)
      # resize_im_shape.append(video_resize.shape[2:4])
      resize_im_shape.append((video_resize.shape[2], video_resize.shape[3]))

    # return FeatureGrids(
    #     tuple(feature_grid), tuple(hires_feats), tuple(highest_feats), tuple(resize_im_shape)
    # )
    return FeatureGrids(
        lowres=feature_grid, hires=hires_feats, highest=highest_feats, resolutions=resize_im_shape
    )

  def estimate_trajectories(
      self,
      video_size: Tuple[int, int],
      is_training: bool,
      feature_grids: FeatureGrids,
      query_features: QueryFeatures,
      query_points_in_video: torch.Tensor,
      query_chunk_size: int,
  ) -> Dict[str, List[torch.Tensor]]:#Mapping[str, Any]:
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
      causal_context: If provided, a dict of causal context to use for
        refinement.
      get_causal_context: If True, return causal context in the output.

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
    del is_training

    # def train2orig(x):
    #   return utils.convert_grid_coordinates(
    #       x,
    #       self.initial_resolution[::-1],
    #       video_size[::-1],
    #       coordinate_format='xy',
    #   )

    pts_iters: List[List[torch.Tensor]] = torch.jit.annotate(List[List[torch.Tensor]], [])
    #new_causal_context: List[List[torch.Tensor]] = torch.jit.annotate(List[List[torch.Tensor]], [])
    num_iters = self.num_pips_iter
    for _ in range(num_iters):
      pts_iters.append([])
      

    num_queries = query_features.lowres_supp[0].shape[1]

    for ch in range(0, num_queries, query_chunk_size):
      
      if query_points_in_video is not None:
        infer_query_points = query_points_in_video[
            :, ch : ch + query_chunk_size
        ]
        num_frames = feature_grids.lowres[0].shape[1]
        infer_query_points = convert_grid_coordinates(
            infer_query_points,
            (num_frames,) + video_size,
            (num_frames,) + self.initial_resolution,
            coordinate_format='tyx',
        )
      else:
        infer_query_points = None

      q_frame = int(infer_query_points[0,0,0].item())
      points = torch.flip(infer_query_points[:, :, 1:], dims=(-1,)).unsqueeze(2).repeat(1, 1, num_frames, 1)
      # points = torch.stack([points[:, :, 1], points[:, :, 0]], dim=-1).unsqueeze(2).repeat(1, 1, num_frames, 1)
      #points = infer_query_points[...,1:][...,[1,0]].unsqueeze(2).repeat(1, 1, num_frames, 1)
      
      #mixer_feats: torch.Tensor = torch.jit.annotate(torch.Tensor, torch.empty(0))
      for i in range(num_iters):
        feature_level = -1
        supports = [
            query_features.hires_supp[feature_level][:, ch:ch + query_chunk_size],
            query_features.lowres_supp[feature_level][:, ch:ch + query_chunk_size],
            query_features.highest_supp[feature_level][:, ch:ch + query_chunk_size],
        ]
        pyramid = [
            feature_grids.hires[feature_level],
            feature_grids.lowres[feature_level],
            feature_grids.highest[feature_level],
        ]
        for _ in range(self.pyramid_level):
          pyramid.append(
              F.avg_pool3d(
                  pyramid[-1],
                  kernel_size=(2, 2, 1),
                  stride=(2, 2, 1),
                  padding=0,
              )
          )
        
        points = self.joint_refine(
            supports,
            pyramid,
            points.detach(),
            orig_hw=self.initial_resolution,
            resize_hw=feature_grids.resolutions[feature_level],
            q_frame=q_frame,
        )
        
        pts_iters[i].append(
          convert_grid_coordinates(
            points,
            self.initial_resolution[::-1],
            video_size[::-1],
            coordinate_format='xy',
          )
        )
       

    points = []
    for i, _ in enumerate(pts_iters):
      points.append(torch.cat(pts_iters[i], dim=1))
      
    out = dict(
        tracks=points,
    )
    return out



  def joint_refine(
        self,
        support_feature: List[torch.Tensor],
        pyramid: List[torch.Tensor],
        pos_guess: torch.Tensor,
        orig_hw: Tuple[int, int],
        resize_hw: Tuple[int, int],
        q_frame: int,
        ) -> torch.Tensor:

        # for tf in target_feature:
        #    print(tf.shape)
        # for sf in support_feature:
        #    print(sf.shape)
        # raise KeyboardInterrupt
        orig_h, orig_w = orig_hw
        resized_h, resized_w = resize_hw
        corrs_pyr: List[torch.Tensor] = []
        #assert len(target_feature) == len(pyramid)
        pyridx:int = 0
        #for pyridx, (query, supp, grid, cmdlayer) in enumerate(zip(target_feature, support_feature, pyramid, self.cmdtop)):
        for pyridx, cmdlayer in enumerate(self.cmdtop):
          supp = support_feature[pyridx]
          grid = pyramid[pyridx]
          #cmdlayer = self.cmdtop[pyridx]  # ModuleList element

          # note: interp needs [y,x]
          coords = convert_grid_coordinates(
              pos_guess, (orig_w, orig_h), grid.shape[-2:-4:-1]
          )
          coords = torch.flip(coords, dims=(-1,))

          support_size = self.patch_size #7
          ctxx, ctxy = torch.meshgrid(
            torch.arange(-(support_size // 2), support_size // 2 + 1), 
            torch.arange(-(support_size // 2), support_size // 2 + 1),
            indexing='xy',
          )
          ctx = torch.stack([ctxy, ctxx], dim=-1)
          ctx = ctx.reshape(-1, 2).to(coords.device)
          coords2 = coords.unsqueeze(3) + ctx.unsqueeze(0).unsqueeze(0).unsqueeze(0)
          neighborhood = map_coordinates_2d(grid, coords2)
          # print(coords.shape, coords2.shape, neighborhood.shape)
          # raise KeyboardInterrupt
          #neighborhood = rearrange(neighborhood, 'b n t (h w) c -> b n t h w c', h=support_size, w=support_size)
          b, n, t, hw, c = neighborhood.shape
          neighborhood = neighborhood.reshape(b, n, t, support_size, support_size, c)
          patches_input = torch.einsum('bnthwc,bnijc->bnthwij', neighborhood, supp)
          #patches_input = rearrange(patches_input, 'b n t h w i j -> (b n t) h w i j')
          b, n, t, h, w, i, j = patches_input.shape
          patches_input = patches_input.reshape(b * n * t, h, w, i, j)
          # patches_emb = self.cmdtop[pyridx](patches_input)
          patches_emb = cmdlayer(patches_input)
          #patches = rearrange(patches_emb, '(b n t) c -> b n t c', b=neighborhood.shape[0], n=neighborhood.shape[1])
          patches = patches_emb.reshape(b, n, t, -1)   

          corrs_pyr.append(patches)
        corrs_pyr = torch.concatenate(corrs_pyr, dim=-1)

        corrs_chunked = corrs_pyr
        pos_guess_input = pos_guess
        
        mlp_input_list = [
            corrs_chunked
        ]

        rel_pos_forward = F.pad(pos_guess_input[..., :-1, :] - pos_guess_input[..., 1:, :], (0, 0, 0, 1))
        rel_pos_backward = F.pad(pos_guess_input[..., 1:, :] - pos_guess_input[..., :-1, :], (0, 0, 1, 0))
        scale = torch.tensor([resized_w / orig_w, resized_h / orig_h]) / torch.tensor([orig_w, orig_h])
        scale = scale.to(pos_guess_input.device)
        rel_pos_forward = rel_pos_forward * scale
        rel_pos_backward = rel_pos_backward * scale
        rel_pos_emb_input = posenc(torch.cat([rel_pos_forward, rel_pos_backward], dim=-1), min_deg=0, max_deg=10) # batch, num_points, num_frames, 84
        mlp_input_list.append(rel_pos_emb_input)
        mlp_input = torch.cat(mlp_input_list, dim=-1)

        #x = rearrange(mlp_input, 'b n f c -> (b n) f c')
        x = mlp_input.reshape(mlp_input.shape[0] * mlp_input.shape[1], mlp_input.shape[2], mlp_input.shape[3])
        res = self.torch_pips_mixer(x)
        # raise KeyboardInterrupt
        #res = rearrange(res, '(b n) f c -> b n f c', b=mlp_input.shape[0])
        res = res.reshape(mlp_input.shape[0], mlp_input.shape[1], mlp_input.shape[2], -1)
        
        pos_update = convert_grid_coordinates(
            res[..., :2],
            (resized_w, resized_h),
            (orig_w, orig_h),
        )
        #pos_update[:,:,q_frame,:] = 0.0
        # print(pos_guess.shape, pos_update.shape, pos_guess[0,0,q_frame], pos_update[0,0,q_frame])
        # raise KeyboardInterrupt
        return pos_update + pos_guess
  
  def interpolate_time_embed(self, x, t):
        previous_dtype = x.dtype
        T = self.time_emb.shape[1]

        if t == T:
            return self.time_emb

        time_emb = self.time_emb.float()
        time_emb = F.interpolate(
            time_emb.permute(0, 2, 1), size=t, mode="linear"
        ).permute(0, 2, 1)
        return time_emb.to(previous_dtype)
