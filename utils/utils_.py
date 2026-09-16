import cv2
import numpy as np
from PIL import Image
import torch
from skimage.color import gray2rgb
from utils import viz_utils
import random
from typing import List, Tuple, Sequence
import torch.nn.functional as F


def local_attn_mask(T: int, W: int) -> torch.Tensor:
    mask = torch.full((T, T), float('-inf'))  # default disallowed
    for i in range(T):
        start = max(0, i - W)
        end = min(T, i + W + 1)
        mask[i, start:end] = 0.0  # allow attention within window
    
    return mask

def map_coordinates_2d(
    feats: torch.Tensor, coordinates: torch.Tensor
) -> torch.Tensor:
  """Maps 2D coordinates to feature maps using bilinear interpolation.

  The function performs bilinear interpolation on the feature maps (`feats`)
  at the specified `coordinates`. The coordinates are normalized between
  -1 and 1 The result is a tensor of sampled features corresponding
  to these coordinates.

  Args:
    feats (Tensor): A 5D tensor of shape (N, T, H, W, C) representing feature
      maps, where N is the batch size, T is the number of frames, H and W are
      height and width, and C is the number of channels.
    coordinates (Tensor): A 5D tensor of shape (N, P, T, S, XY) representing
      coordinates, where N is the batch size, P is the number of points, T is
      the number of frames, S is the number of samples, and XY represents the 2D
      coordinates.

  Returns:
    Tensor: A 5D tensor of the sampled features corresponding to the
      given coordinates, of shape (N, P, T, S, C).
  """
  n, t, h, w, c = feats.shape
  x = feats.permute(0, 1, 4, 2, 3).view(n * t, c, h, w)

  n, p, t, s, xy = coordinates.shape
  y = coordinates.permute(0, 2, 1, 3, 4).reshape(n * t, p, s, xy)
  y = 2 * (y / torch.tensor([h, w], device=feats.device)) - 1
  y = torch.flip(y, dims=(-1,)).float()

  out = F.grid_sample(
      x, y, mode='bilinear', align_corners=False, padding_mode='zeros'
  )
  _, c, _, _ = out.shape
  out = out.permute(0, 2, 3, 1).view(n, t, p, s, c).permute(0, 2, 1, 3, 4)

  return out

def generate_default_resolutions(full_size, train_size, num_levels=None):
  """Generate a list of logarithmically-spaced resolutions.

  Generated resolutions are between train_size and full_size, inclusive, with
  num_levels different resolutions total.  Useful for generating the input to
  refinement_resolutions in PIPs.

  Args:
    full_size: 2-tuple of ints.  The full image size desired.
    train_size: 2-tuple of ints.  The smallest refinement level.  Should
      typically match the training resolution, which is (256, 256) for TAPIR.
    num_levels: number of levels.  Typically each resolution should be less than
      twice the size of prior resolutions.

  Returns:
    A list of resolutions.
  """
  if all([x == y for x, y in zip(train_size, full_size)]):
    return [train_size]

  if num_levels is None:
    size_ratio = np.array(full_size) / np.array(train_size)
    num_levels = int(np.ceil(np.max(np.log2(size_ratio))) + 1)

  if num_levels <= 1:
    return [train_size]

  h, w = full_size[0:2]
  if h % 8 != 0 or w % 8 != 0:
    print(
        'Warning: output size is not a multiple of 8. Final layer '
        + 'will round size down.'
    )
  ll_h, ll_w = train_size[0:2]

  sizes = []
  for i in range(num_levels):
    size = (
        int(round((ll_h * (h / ll_h) ** (i / (num_levels - 1))) // 8)) * 8,
        int(round((ll_w * (w / ll_w) ** (i / (num_levels - 1))) // 8)) * 8,
    )
    sizes.append(size)
  return sizes


def get_resized_frames(frames, height, width):
  """Resize frames to the given width and height

  Args:
      frames (ndarray): [S, H, W, C]
      height (int): target height
      width (int): target width

  Returns:
      resized_frames (ndarray): [S, height, width, C]
  """
  resized_frames = []

  for frame in frames:
      # Resize the frame
      resized_frame = cv2.resize(frame, (width, height))
      # Append the resized frame to the list
      resized_frames.append(resized_frame)

  # Convert the list of frames to a NumPy array
  resized_frames = np.array(resized_frames)

  if len(resized_frames.shape) < 4:
     resized_frames = resized_frames[..., np.newaxis]

  return resized_frames

def soft_argmax_heatmap_batched(softmax_val, threshold=5):
  """Test if two image resolutions are the same."""
  b, h, w, d1, d2 = softmax_val.shape
  y, x = torch.meshgrid(
      torch.arange(d1, device=softmax_val.device),
      torch.arange(d2, device=softmax_val.device),
      indexing='ij',
  )
  coords = torch.stack([x + 0.5, y + 0.5], dim=-1).to(softmax_val.device)
  softmax_val_flat = softmax_val.reshape(b, h, w, -1)
  argmax_pos = torch.argmax(softmax_val_flat, dim=-1)

  pos = coords.reshape(-1, 2)[argmax_pos]
  valid = (
      torch.sum(
          torch.square(
              coords[None, None, None, :, :, :] - pos[:, :, :, None, None, :]
          ),
          dim=-1,
          keepdims=True,
      )
      < threshold**2
  )

  weighted_sum = torch.sum(
      coords[None, None, None, :, :, :]
      * valid
      * softmax_val[:, :, :, :, :, None],
      dim=(3, 4),
  )
  sum_of_weights = torch.maximum(
      torch.sum(valid * softmax_val[:, :, :, :, :, None], dim=(3, 4)),
      torch.tensor(1e-12, device=softmax_val.device),
  )
  return weighted_sum / sum_of_weights


def heatmaps_to_points(
    all_pairs_softmax,
    image_shape,
    threshold=5,
    query_points=None,
):
  """Convert heatmaps to points using soft argmax."""

  out_points = soft_argmax_heatmap_batched(all_pairs_softmax, threshold)
  feature_grid_shape = all_pairs_softmax.shape[1:]
  # Note: out_points is now [x, y]; we need to divide by [width, height].
  # image_shape[3] is width and image_shape[2] is height.
  out_points = convert_grid_coordinates(
      out_points,#.detach(),
      feature_grid_shape[3:1:-1],
      image_shape[3:1:-1],
  )
  assert feature_grid_shape[1] == image_shape[1]
  if query_points is not None:
    # The [..., 0:1] is because we only care about the frame index.
    query_frame = convert_grid_coordinates(
        query_points,#.detach(),
        image_shape[1:4],
        feature_grid_shape[1:4],
        coordinate_format='tyx',
    )[..., 0:1]

    query_frame = torch.round(query_frame)
    frame_indices = torch.arange(image_shape[1], device=query_frame.device)[
        None, None, :
    ]
    is_query_point = query_frame == frame_indices

    is_query_point = is_query_point[:, :, :, None]
    out_points = (
        out_points * ~is_query_point
        + torch.flip(query_points[:, :, None], dims=(-1,))[..., 0:2]
        * is_query_point
    )

  return out_points


def bilinear(x: torch.Tensor, resolution: tuple[int, int]) -> torch.Tensor:
  """Resizes a 5D tensor using bilinear interpolation.

  Args:
        x: A 5D tensor of shape (B, T, W, H, C) where B is batch size, T is
          time, W is width, H is height, and C is the number of channels.
    resolution: The target resolution as a tuple (new_width, new_height).

  Returns:
    The resized tensor.
  """
  b, t, h, w, c = x.size()
  x = x.permute(0, 1, 4, 2, 3).reshape(b, t * c, h, w)
  x = F.interpolate(x, size=resolution, mode='bilinear', align_corners=False)
  b, _, h, w = x.size()
  x = x.reshape(b, t, c, h, w).permute(0, 1, 3, 4, 2)
  return x


def is_same_res(r1, r2):
  """Test if two image resolutions are the same."""
  return all([x == y for x, y in zip(r1, r2)])


def map_coordinates_3d(
    feats: torch.Tensor, coordinates: torch.Tensor
) -> torch.Tensor:
  """Maps 3D coordinates to corresponding features using bilinear interpolation.

  Args:
    feats: A 5D tensor of features with shape (B, W, H, D, C), where B is batch
      size, W is width, H is height, D is depth, and C is the number of
      channels.
    coordinates: A 3D tensor of coordinates with shape (B, N, 3), where N is the
      number of coordinates and the last dimension represents (W, H, D)
      coordinates.

  Returns:
    The mapped features tensor.
  """
  x = feats.permute(0, 4, 1, 2, 3)
  y = coordinates[:, :, None, None, :].float()
  y[..., 0] += 0.5
  y = 2 * (y / torch.tensor(x.shape[2:], device=y.device)) - 1
  y = torch.flip(y, dims=(-1,))
  out = (
      F.grid_sample(
          x, y, mode='bilinear', align_corners=False, padding_mode='border'
      )
      .squeeze(dim=(3, 4))
      .permute(0, 2, 1)
  )
  return out

def convert_grid_coordinates(
    coords: torch.Tensor,
    input_grid_size: Sequence[int],
    output_grid_size: Sequence[int],
    coordinate_format: str = 'xy',
) -> torch.Tensor:
  """Convert grid coordinates to correct format."""
  if isinstance(input_grid_size, tuple):
    input_grid_size = torch.tensor(input_grid_size, device=coords.device)
  if isinstance(output_grid_size, tuple):
    output_grid_size = torch.tensor(output_grid_size, device=coords.device)

  if coordinate_format == 'xy':
    if input_grid_size.shape[0] != 2 or output_grid_size.shape[0] != 2:
      raise ValueError(
          'If coordinate_format is xy, the shapes must be length 2.'
      )
  elif coordinate_format == 'tyx':
    if input_grid_size.shape[0] != 3 or output_grid_size.shape[0] != 3:
      raise ValueError(
          'If coordinate_format is tyx, the shapes must be length 3.'
      )
    if input_grid_size[0] != output_grid_size[0]:
      raise ValueError('converting frame count is not supported.')
  else:
    raise ValueError('Recognized coordinate formats are xy and tyx.')

  position_in_grid = coords
  position_in_grid = position_in_grid * output_grid_size / input_grid_size

  return position_in_grid


def bilinear_sampler(img, coords, mode:str='bilinear', mask:bool=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    # go to 0,1 then 0,2 then -1,1
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    return img


def requires_grad(parameters, flag=True, init_stage=True):
    for p in parameters:
        p.requires_grad = flag
        
   

def fetch_optimizer(lr, wdecay, epsilon, epochs, steps_per_epoch, params):
    #optimizer = ADOPT(params, lr=lr, weight_decay=wdecay, eps=epsilon, decoupled=True)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wdecay, eps=epsilon)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
       optimizer, lr, epochs=epochs, steps_per_epoch=steps_per_epoch, pct_start=0.1, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler



def reduce_masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: Tuple[int, int], keepdim: bool =False):
    # x and mask are the same shape, or at least broadcastably so < actually it's safer if you disallow broadcasting
    # returns shape-1
    # axis can be a list of axes
    for (a,b) in zip(x.size(), mask.size()):
        # if not b==1: 
        assert(a==b) # some shape mismatch!
    # assert(x.size() == mask.size())
    prod = x*mask
    # if dim is None:
    #     numer = torch.sum(prod)
    #     denom = EPS+torch.sum(mask)
    # else:
    numer = torch.sum(prod, dim=dim, keepdim=keepdim)
    denom = 1e-6+torch.sum(mask, dim=dim, keepdim=keepdim)
    mean = numer/denom
    # print(x.shape, mask.shape, prod.shape, numer.shape, denom.shape, mean.shape)
    # raise KeyboardInterrupt
    # print(numer.shape, denom.shape, mean.shape)
    # raise KeyboardInterrupt
    return mean

def sequence_loss(flow_preds: List[torch.Tensor], flow_gt: torch.Tensor, vis: torch.Tensor, valids: torch.Tensor, gamma: float = 0.8):
    """ Loss function defined over sequence of flow predictions """
    B, S, N, D = flow_gt.shape
    assert(D==2)
    B, S1, N = vis.shape
    B, S2, N = valids.shape
    assert(S==S1)
    assert(S==S2)
    n_predictions = len(flow_preds)    
    flow_loss = 0.0
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        flow_pred = flow_preds[i]
        i_loss = (flow_pred - flow_gt).abs() # B,S,N,2
        i_loss = torch.mean(i_loss, dim=3) # B,S,N
        #flow_loss += i_weight * basic.reduce_masked_mean(i_loss, valids)
        flow_loss += i_weight * reduce_masked_mean(i_loss, valids, dim=(1, 2), keepdim=True).squeeze(1)
    flow_loss = flow_loss/n_predictions
    return flow_loss

def points_to_tensor(points: list, qt: int, orig_H: int, orig_W: int, target: int = 256) -> torch.Tensor:
    """
    Convert [(x1,y1), ..., (xn,yn)] to tensor of shape [1, n, 3]
    where last dim is (qt, x, y), with x/y scaled to target resolution.

    Args:
        points  : list of (x, y) tuples or np.array([x, y])
        qt      : single int, same for all points
        orig_H  : original frame height
        orig_W  : original frame width
        target  : target resolution (default 256)

    Returns:
        tensor of shape [1, n, 3], dtype float32
    """
    scale_x = target / orig_W
    scale_y = target / orig_H

    arr = np.array(
        [[qt, p[0] * scale_x, p[1] * scale_y] for p in points],
        dtype=np.float32
    )  # (n, 3)

    return torch.tensor(arr).unsqueeze(0)  # (1, n, 3)

def randomize_points(points, fix=None):
  """sample points from random frames.

  Args:
      points (_type_): [B, N, S, 2], (x, y) -> [0.0 - 1.0]
      
  Returns:
      _type_: [B, N, 3] -> (float, float, float)
  """    
  B, N, S, _ = points.shape
  output = torch.zeros((B, N, 3), dtype=torch.float)

  for i in range(N):
    frame = random.randint(0, S-1) if fix is None else fix  # Randomly selected frame index
    output[:,i,0] = frame
    output[:,i,1:] = points[:,i, frame]

  return output


def getMetricsDict(framewisw=False, T=100):
    metrics = {
        'occlusion_accuracy': np.zeros((T,), dtype=float) if framewisw else 0.0,
        'pts_within_1': np.zeros((T,), dtype=float) if framewisw else 0.0,
        'pts_within_2': np.zeros((T,), dtype=float) if framewisw else 0.0,
        'pts_within_4': np.zeros((T,), dtype=float) if framewisw else 0.0,
        'pts_within_8': np.zeros((T,), dtype=float) if framewisw else 0.0,
        'pts_within_16': np.zeros((T,), dtype=float) if framewisw else 0.0,
        'average_pts_within_thresh': np.zeros((T,), dtype=float) if framewisw else 0.0,
        'jaccard_1': np.zeros((T,), dtype=float) if framewisw else 0.0,
        'jaccard_2': np.zeros((T,), dtype=float) if framewisw else 0.0,
        'jaccard_4': np.zeros((T,), dtype=float) if framewisw else 0.0,
        'jaccard_8': np.zeros((T,), dtype=float) if framewisw else 0.0,
        'jaccard_16': np.zeros((T,), dtype=float) if framewisw else 0.0,
        'average_jaccard': np.zeros((T,), dtype=float) if framewisw else 0.0,
        'inference_time': 0.0,
        'peak_memory': 0.0,
        'survival': np.zeros((T,), dtype=float) if framewisw else 0.0,
        'median_traj_error': np.zeros((T,), dtype=float) if framewisw else 0.0
    }
    return metrics


def concatenate_videos_in_grid(videos, grid_shape=None):
    """
    Concatenates multiple videos into a single grid video without resizing individual videos.
    Pads videos to fit the maximum frame length, height, and width.

    Args:
        videos (list of np.ndarray): List of videos, each of shape (T, H, W, C).
        grid_shape (tuple, optional): Desired grid layout (rows, cols). 
                                      If None, it will auto-calculate a square layout.

    Returns:
        np.ndarray: Concatenated video grid of shape (T, total_height, total_width, C).
    """
    n_frames = videos[0].shape[0]
    height = videos[0].shape[1]
    width = videos[0].shape[2]
    channels = videos[0].shape[3]

    # Determine grid shape if not provided
    num_videos = len(videos)
    if grid_shape is None:
        grid_rows = int(np.sqrt(num_videos))
        grid_cols = int(np.ceil(num_videos / grid_rows))
    else:
        grid_rows, grid_cols = grid_shape

    # Calculate final video dimensions
    total_height = grid_rows * height
    total_width = grid_cols * width

    # Initialize the final concatenated video grid
    video_grid = np.zeros((n_frames, total_height, total_width, channels), dtype=videos[0].dtype)

    # Place each video in its grid position
    for idx, video in enumerate(videos):
        row = idx // grid_cols
        col = idx % grid_cols

        # Get the starting coordinates for the current video's position
        start_y = row * height
        start_x = col * width

        # Copy the video frames into the correct grid position with padding if necessary
        video_grid[:n_frames, start_y:start_y + height, start_x:start_x + width, :] = video

    return video_grid

def add_text_to_frames(frames, text, position=(5, 25), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=4):
    frames_with_text = []
    for frame in frames:
        # Convert frame to BGR if it's in grayscale
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Add text to the frame
        frame_with_text = frame.copy()
        cv2.putText(frame_with_text, text, position, font, font_scale, color, thickness)
        frames_with_text.append(frame_with_text)
    
    return np.stack(frames_with_text)


def paint_vid(frames, points, visibs=None, gray = False, colormap=None):
    """This will paint the points into the frames.

    Args:
        frames (ndarray): (n_frames, width, height, channel)
        points (ndarray): (n_points, n_frames, 2)
        occluded (ndarray): (n_points, n_frames)
    
    Return:
        painted_frames (ndarray): (n_frames, width, height, channel)
    """
    # if visibs == None:
    #    visibs = np.ones(points.shape[:2])
    
    
    if gray:
       frames = frames.squeeze()
       frames = gray2rgb(frames)

    scale_factor = np.array(frames.shape[2:0:-1])[np.newaxis, np.newaxis, :]
    painted_frames = viz_utils.paint_point_track(
        frames,
        points  * scale_factor,
        visibs,
        colormap,
    )

    return painted_frames


def bilinear_sample2d(im, x, y, return_inbounds:bool=False):
    # x and y are each B, N
    # output is B, C, N
    B, C, H, W = im.shape
    N = x.shape[1]

    x = x.float()
    y = y.float()
    H_f = torch.tensor(H, dtype=torch.float32, device=x.device)
    W_f = torch.tensor(W, dtype=torch.float32, device=x.device)
    
    max_y = (H_f - 1).int()
    max_x = (W_f - 1).int()

    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1
    
    x0_clip = torch.clamp(x0, 0, max_x)
    x1_clip = torch.clamp(x1, 0, max_x)
    y0_clip = torch.clamp(y0, 0, max_y)
    y1_clip = torch.clamp(y1, 0, max_y)
    
    dim2 = W
    dim1 = W * H

    base = torch.arange(0, B, dtype=torch.int64, device=x.device) * dim1
    base = base.view(B, 1).expand(B, N)

    base_y0 = base + y0_clip * dim2
    base_y1 = base + y1_clip * dim2

    idx_y0_x0 = base_y0 + x0_clip
    idx_y0_x1 = base_y0 + x1_clip
    idx_y1_x0 = base_y1 + x0_clip
    idx_y1_x1 = base_y1 + x1_clip

    # Use the indices to lookup pixels in the flat image
    im_flat = im.permute(0, 2, 3, 1).reshape(B * H * W, C)
    
    i_y0_x0 = im_flat[idx_y0_x0.long()]
    i_y0_x1 = im_flat[idx_y0_x1.long()]
    i_y1_x0 = im_flat[idx_y1_x0.long()]
    i_y1_x1 = im_flat[idx_y1_x1.long()]

    # Calculate interpolated values
    x0_f = x0.float()
    x1_f = x1.float()
    y0_f = y0.float()
    y1_f = y1.float()

    w_y0_x0 = ((x1_f - x) * (y1_f - y)).unsqueeze(2)
    w_y0_x1 = ((x - x0_f) * (y1_f - y)).unsqueeze(2)
    w_y1_x0 = ((x1_f - x) * (y - y0_f)).unsqueeze(2)
    w_y1_x1 = ((x - x0_f) * (y - y0_f)).unsqueeze(2)

    output = w_y0_x0 * i_y0_x0 + w_y0_x1 * i_y0_x1 + \
             w_y1_x0 * i_y1_x0 + w_y1_x1 * i_y1_x1
    # output is B*N x C
    output = output.view(B, N, C)
    output = output.permute(0, 2, 1)
    # output is B x C x N

    return output  # B, C, N


def posemb_sincos_2d_xy(xy:torch.Tensor=torch.empty(0), C:int=0, temperature:float=10000, cat_coords: bool=False):
    device = xy.device
    dtype = xy.dtype
    B, S, D = xy.shape
    assert(D==2)
    x = xy[:,:,0]
    y = xy[:,:,1]
    assert (C % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(C // 4, device=device) / (C // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1) 
    pe = pe.reshape(B,S,C).type(dtype)
    if cat_coords:
        pe = torch.cat([pe, xy], dim=2) # B,N,C+2
    return pe

def play_video(video_file, target_size=(256, 256)):
    
    # Open the video file
    cap = cv2.VideoCapture(video_file)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None, None
    

    # List to store frames for the GIF
    rgbs = []
    gframes = []
    new_frame = True
    text = "Press 'q' on the video player to quit the video player after finishing one cycle."
    print(text)
    while True:
        #Restart the video from the beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            #Read a frame from the video
            ret, frame = cap.read()

            # Break the inner loop if we have reached the end of the video
            if not ret:
                new_frame = False
                break
            
            # # Add text to the frame
            # cv.putText(frame, text, (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # Resize the grayscale frame to the target size
            frame = cv2.resize(frame, target_size)
            # Display the frame
            cv2.imshow('Video Player', frame)

            # Append the frame to the list if it is new
            if new_frame == True:
              gframes.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
              rgbs.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    
            #Check for user input to quit (press 'q' key)
            key = cv2.waitKey(30)
            if key == ord('q'):
                # Release the video capture object and close the display window
                cap.release()
                cv2.destroyAllWindows()

                # # Save the video as a gif file
                # output_gif_file = "results/input_video.gif" 
                # frames[0].save(output_gif_file, save_all=True, append_images=frames[1:], loop=0)
                # print(f"Input video is saved as a GIF file to {output_gif_file}")
                # Convert the list of frames into a NumPy array
                rgbs = np.array(rgbs)
                gframes = np.array(gframes)
                # if len(frames.shape) < 4:
                #     frames = frames[..., np.newaxis s]
                
                return rgbs, gframes