import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Sequence, Tuple, NamedTuple, Union, Optional
from einops import rearrange
import math
from utils.utils_ import posemb_sincos_2d_xy, bilinear_sampler, local_attn_mask

@torch.jit.script
def get_relative_positions(seq_len: int, reverse: bool = False, device: Optional[torch.device] = None) -> torch.Tensor:
    x = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, T]
    y = torch.arange(seq_len, device=device).unsqueeze(1)  # [T, 1]
    if reverse:
        return torch.triu(y - x)
    else:
        return torch.tril(x - y)

@torch.jit.script
def get_alibi_slope(num_heads: int, device: Optional[torch.device] = None) -> torch.Tensor:
    base = torch.tensor(24.0, device=device).pow(1.0 / float(num_heads))
    vals = []
    for i in range(num_heads):
        vals.append((1.0 / base.pow(float(i + 1))))
    return torch.stack(vals).to(dtype=torch.float32, device=device).view(-1, 1, 1)

class MultiHeadAttention(nn.Module):
    """TorchScript-compatible Multi-headed attention (MHA)."""

    def __init__(
        self,
        num_heads: int,
        key_size: int,
        value_size: Optional[int] = None,
        model_size: Optional[int] = None,
        with_bias: bool = True,
    ):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size if value_size is not None else key_size
        self.model_size = model_size if model_size is not None else key_size * num_heads

        self.with_bias = with_bias

        self.query_proj = nn.Linear(num_heads * key_size, num_heads * key_size, bias=with_bias)
        self.key_proj = nn.Linear(num_heads * key_size, num_heads * key_size, bias=with_bias)
        self.value_proj = nn.Linear(num_heads * self.value_size, num_heads * self.value_size, bias=with_bias)
        self.final_proj = nn.Linear(num_heads * self.value_size, self.model_size, bias=with_bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = query.size()

        # query_heads = self._linear_projection(query, self.key_size, self.query_proj)
        # key_heads = self._linear_projection(key, self.key_size, self.key_proj)
        # value_heads = self._linear_projection(value, self.value_size, self.value_proj)

        query_heads = self._linear_projection(query, self.key_size, 'query')
        key_heads = self._linear_projection(key, self.key_size, 'key')
        value_heads = self._linear_projection(value, self.value_size, 'value')

        device = query.device

        if mask is None:
          # Forward bias
          bias_forward = get_alibi_slope(self.num_heads // 2, device) * get_relative_positions(sequence_length, False, device)
          bias_forward = bias_forward + torch.triu(torch.full_like(bias_forward, -1e9), diagonal=1)

          # Backward bias
          bias_backward = get_alibi_slope(self.num_heads // 2, device) * get_relative_positions(sequence_length, True, device)
          bias_backward = bias_backward + torch.tril(torch.full_like(bias_backward, -1e9), diagonal=-1)

          attn_bias = torch.cat([bias_forward, bias_backward], dim=0)
        else:
          attn_bias = mask
        # print(attn_bias.shape, query_heads.shape)
        # print(attn_bias[0])
        # raise KeyboardInterrupt

        scale = 1.0 / torch.sqrt(torch.tensor(float(self.key_size), device=device))

        attn = F.scaled_dot_product_attention(
            query_heads, key_heads, value_heads, attn_mask=attn_bias, scale=scale
        )
        attn = attn.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, -1)

        return self.final_proj(attn)

    def _linear_projection(self, x: torch.Tensor, head_size: int, proj_type: str) -> torch.Tensor:
        if proj_type == "query":
            y = self.query_proj(x)
        elif proj_type == "key":
            y = self.key_proj(x)
        else:
            y = self.value_proj(x)

        batch_size, sequence_length, _ = x.shape
        return y.reshape((batch_size, sequence_length, self.num_heads, head_size)).permute(0, 2, 1, 3)

class TransformerLayer(nn.Module):
    def __init__(self, num_heads: int, attn_size: int, widening_factor: int, dropout_rate: float, sp_attn: bool = False):
        super(TransformerLayer, self).__init__()
        self.sp_attn = sp_attn
        self.attn = MultiHeadAttention(num_heads, attn_size, model_size=attn_size * num_heads)
        if sp_attn:
           self.sp_attn = MultiHeadAttention(num_heads, attn_size, model_size=attn_size * num_heads)
        self.dense = nn.Sequential(
            nn.Linear(attn_size * num_heads, widening_factor * attn_size * num_heads),
            nn.GELU(),
            nn.Linear(widening_factor * attn_size * num_heads, attn_size * num_heads)
        )
        self.layer_norm1 = nn.LayerNorm(attn_size * num_heads)
        self.layer_norm2 = nn.LayerNorm(attn_size * num_heads)
        self.dropout_rate = dropout_rate

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, sp_window: Optional[int] = 10) -> torch.Tensor:
        h_norm = self.layer_norm1(x)
        h_attn = self.attn(h_norm, h_norm, h_norm, mask)
        h_attn = F.dropout(h_attn, p=self.dropout_rate, training=self.training)
        if self.sp_attn:
          # print(x.shape, h_norm.shape)
          # raise KeyboardInterrupt
          s_norm = h_norm.permute(1, 0, 2).contiguous()
          N = s_norm.shape[1]
          mask = local_attn_mask(N, sp_window).to(x.device)
          s_attn = self.sp_attn(s_norm, s_norm, s_norm, mask)
          s_attn = F.dropout(s_attn, p=self.dropout_rate, training=self.training)
          h = x + h_attn + s_attn.permute(1, 0, 2).contiguous()
        else:
          h = x + h_attn

        h_norm = self.layer_norm2(h)
        h_dense = self.dense(h_norm)
        h_dense = F.dropout(h_dense, p=self.dropout_rate, training=self.training)
        return h + h_dense


class Transformer(nn.Module):
    def __init__(self, num_heads: int, num_layers: int, attn_size: int, dropout_rate: float, widening_factor: int = 4, sp_attn: bool = False):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(num_heads, attn_size, widening_factor, dropout_rate, sp_attn=sp_attn)
            for _ in range(num_layers)
        ])
        self.ln_out = nn.LayerNorm(attn_size * num_heads)

    def forward(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = embeddings
        for layer in self.layers:
            h = layer(h, mask)
        return self.ln_out(h)


class KNPTransformer(nn.Module):
    def __init__(self, input_channels, output_channels, dim=512, num_heads=8, num_layers=1, sp_attn:bool = False):
        super(KNPTransformer, self).__init__()
        self.dim = dim

        self.transformer = Transformer(
            num_heads=num_heads,
            num_layers=num_layers,
            attn_size=dim // num_heads,
            dropout_rate=0.,
            widening_factor=4,
            sp_attn=sp_attn
        )
        self.input_proj = nn.Linear(input_channels, dim)
        self.output_proj = nn.Linear(dim, output_channels)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x, mask=None)
        return self.output_proj(x)

class ExtraConvBlock_echo2(nn.Module):
  """Additional convolution block."""

  def __init__(
      self,
      channel_dim,
      channel_multiplier,
  ):
    super().__init__()
    self.channel_dim = channel_dim
    self.channel_multiplier = channel_multiplier

    self.layer_norm = nn.LayerNorm(
        normalized_shape=channel_dim, elementwise_affine=True, bias=True
    )
    self.conv = nn.Conv2d(
        self.channel_dim * 3,
        self.channel_dim * self.channel_multiplier,
        kernel_size=3,
        stride=1,
        padding=1,
    )
    self.conv_1 = nn.Conv2d(
        self.channel_dim * self.channel_multiplier,
        self.channel_dim,
        kernel_size=3,
        stride=1,
        padding=1,
    )

  def forward(self, x):
    x = self.layer_norm(x)
    x = x.permute(0, 3, 1, 2)
    prev_frame = torch.cat([x[0:1], x[:-1]], dim=0)
    next_frame = torch.cat([x[1:], x[-1:]], dim=0)
    resid = torch.cat([x, prev_frame, next_frame], dim=1)
    #print(x.shape, prev_frame.shape, next_frame.shape, resid.shape)
    #raise KeyboardInterrupt
    resid = self.conv(resid)
    resid = F.gelu(resid, approximate='tanh')
    x += self.conv_1(resid)
    x = x.permute(0, 2, 3, 1)
    return x


class ExtraConvs_echo2(nn.Module):
  """Additional CNN."""

  def __init__(
      self,
      num_layers=5,
      channel_dim=256,
      channel_multiplier=4,
  ):
    super().__init__()
    self.num_layers = num_layers
    self.channel_dim = channel_dim
    self.channel_multiplier = channel_multiplier

    self.blocks = nn.ModuleList()
    for _ in range(self.num_layers):
      self.blocks.append(
          ExtraConvBlock_echo2(self.channel_dim, self.channel_multiplier)
      )
    #self.upsampler = CARAFE(in_channels=channel_dim, out_channels=int(channel_dim/2), scale=2)
    

  def forward(self, x):
    # print(x.shape)
    out = []
    for y in x:
      for block in self.blocks:
        y = block(y)
        # print(y.shape)
      out.append(y.unsqueeze(0))
      #print(x.shape)
    #x = self.upsampler(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    #print("\n")
    out = torch.cat(out, dim=0)
    # print(out.shape)
    # raise KeyboardInterrupt
    return out


class ResNet_echo2(nn.Module):
  """ResNet model."""

  def __init__(
      self,
      blocks_per_group: Sequence[int],
      channels_per_group: Sequence[int] = (64, 128, 256, 512),
      use_projection: Sequence[bool] = (True, True, True, True),
      strides: Sequence[int] = (1, 2, 2, 2),
      temp_adapter: bool = False,
  ):
    """Initializes a ResNet model with customizable layers and configurations.

    This constructor allows defining the architecture of a ResNet model by
    setting the number of blocks, channels, projection usage, and strides for
    each group of blocks within the network. It provides flexibility in
    creating various ResNet configurations.

    Args:
      blocks_per_group: A sequence of 4 integers, each indicating the number
        of residual blocks in each group.
      channels_per_group: A sequence of 4 integers, each specifying the number
        of output channels for the blocks in each group. Defaults to (64, 128,
        256, 512).
      use_projection: A sequence of 4 booleans, each indicating whether to use
        a projection shortcut (True) or an identity shortcut (False) in each
        group. Defaults to (True, True, True, True).
      strides: A sequence of 4 integers, each specifying the stride size for
        the convolutions in each group. Defaults to (1, 2, 2, 2).

    The ResNet model created will have 4 groups, with each group's
    architecture defined by the corresponding elements in these sequences.
    """
    super().__init__()
    self.temp_adapter = temp_adapter

    self.initial_conv = Conv2dSamePadding(
        in_channels=3,
        out_channels=channels_per_group[0],
        kernel_size=(7, 7),
        stride=2,
        padding=0,
        bias=False,
    )

    block_groups = []
    for i, _ in enumerate(strides):
      block_groups.append(
          BlockGroup_loco(
              channels_in=channels_per_group[i - 1] if i > 0 else 64,
              channels_out=channels_per_group[i],
              num_blocks=blocks_per_group[i],
              stride=strides[i],
              use_projection=use_projection[i],
          )
      )
    self.block_groups = nn.ModuleList(block_groups)

    if temp_adapter:
      # self.adapter = build_temp_adapter(
      #   input_channels= channels_per_group[:-1], 
      #   intermed_channels=channels_per_group[:-1],
      #   num_layers=len(blocks_per_group) - 1,
      #   num_heads=4,
      #   window_size=13,
      # )
      self.adapter = nn.ModuleList([
         ExtraConvs_echo2(channel_dim=channels_per_group[i], num_layers=1, channel_multiplier=1) for i in range(3) 
      ])

  def forward(self, inputs):
    #b, t, c, h, w = inputs.shape
    result = {}
    #print(inputs.shape)
    out = inputs#.reshape(b*t, c, h, w)
    out = self.initial_conv(out)
    result['initial_conv'] = out
    #print(out.shape)
    for block_id, block_group in enumerate(self.block_groups):
      out = block_group(out)
      #print(out.shape)
      if self.temp_adapter and block_id < len(self.block_groups) - 1:
        # out = self.adapter[block_id](out.unsqueeze(0).permute(0,2,1,3,4))#for BMHA
        #print(out.unsqueeze(0).permute(0,1,3,4,2).shape)
        # raise KeyboardInterrupt
        out = self.adapter[block_id](out.unsqueeze(0).permute(0,1,3,4,2)).squeeze(0).permute(0,3,1,2)# for iTSM
        # print(out.shape)
        # raise KeyboardInterrupt
      result[f'resnet_unit_{block_id}'] = out
    #raise KeyboardInterrupt
    #   print(f'resnet_unit_{block_id}_shape: {out.shape}')
    return result


class ExtraConvBlock_loco(nn.Module):
  """Additional convolution block."""

  def __init__(
      self,
      channel_dim,
      channel_multiplier,
  ):
    super().__init__()
    self.channel_dim = channel_dim
    self.channel_multiplier = channel_multiplier

    self.layer_norm = nn.LayerNorm(
        normalized_shape=channel_dim, elementwise_affine=True, bias=True
    )
    self.conv = nn.Conv2d(
        self.channel_dim,
        self.channel_dim * self.channel_multiplier,
        kernel_size=3,
        stride=1,
        padding=1,
    )
    self.conv_1 = nn.Conv2d(
        self.channel_dim * self.channel_multiplier,
        self.channel_dim,
        kernel_size=3,
        stride=1,
        padding=1,
    )

  def forward(self, x):
    x = self.layer_norm(x)
    x = x.permute(0, 3, 1, 2)
    res = self.conv(x)
    res = F.gelu(res, approximate='tanh')
    x = x + self.conv_1(res)
    x = x.permute(0, 2, 3, 1)
    return x
  
class ExtraConvs_loco(nn.Module):
  """Additional CNN."""

  def __init__(
      self,
      num_layers=5,
      channel_dim=256,
      channel_multiplier=4,
  ):
    super().__init__()
    self.num_layers = num_layers
    self.channel_dim = channel_dim
    self.channel_multiplier = channel_multiplier

    self.blocks = nn.ModuleList()
    for _ in range(self.num_layers):
      self.blocks.append(
          ExtraConvBlock_loco(self.channel_dim, self.channel_multiplier)
      )

  def forward(self, x):
    for block in self.blocks:
      x = block(x)

    return x

class BlockV2_loco(nn.Module):
  """ResNet V2 block."""

  def __init__(
      self,
      channels_in: int,
      channels_out: int,
      stride: Union[int, Sequence[int]],
      use_projection: bool,
  ):
    super().__init__()
    self.padding = (1, 1, 1, 1)
    # Handle assymetric padding created by padding="SAME" in JAX/LAX
    if stride == 1:
      self.padding = (1, 1, 1, 1)
    elif stride == 2:
      self.padding = (0, 2, 0, 2)
    else:
      raise ValueError(
          'Check correct padding using padtype_to_padsin jax._src.lax.lax'
      )

    self.use_projection = use_projection
    if self.use_projection:
      self.proj_conv = Conv2dSamePadding(
          in_channels=channels_in,
          out_channels=channels_out,
          kernel_size=1,
          stride=stride,
          padding=0,
          bias=False,
      )

    self.bn_0 = nn.InstanceNorm2d(
        num_features=channels_in,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=False,
    )
    self.conv_0 = Conv2dSamePadding(
        in_channels=channels_in,
        out_channels=channels_out,
        kernel_size=3,
        stride=stride,
        padding=0,
        bias=False,
    )

    self.conv_1 = Conv2dSamePadding(
        in_channels=channels_out,
        out_channels=channels_out,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    self.bn_1 = nn.InstanceNorm2d(
        num_features=channels_out,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=False,
    )

  def forward(self, inputs):
    x = shortcut = inputs

    x = self.bn_0(x)
    x = torch.relu(x)
    if self.use_projection:
      shortcut = self.proj_conv(x)

    x = self.conv_0(x)

    x = self.bn_1(x)
    x = torch.relu(x)
    # no issues with padding here as this layer always has stride 1
    x = self.conv_1(x)

    return x + shortcut
  
class BlockGroup_loco(nn.Module):
  """Higher level block for ResNet implementation."""

  def __init__(
      self,
      channels_in: int,
      channels_out: int,
      num_blocks: int,
      stride: Union[int, Sequence[int]],
      use_projection: bool,
  ):
    super().__init__()
    blocks = []
    for i in range(num_blocks):
      blocks.append(
          BlockV2_loco(
              channels_in=channels_in if i == 0 else channels_out,
              channels_out=channels_out,
              stride=(1 if i else stride),
              use_projection=(i == 0 and use_projection),
          )
      )
    self.blocks = nn.ModuleList(blocks)

  def forward(self, inputs):
    out = inputs
    for block in self.blocks:
      out = block(out)
    return out

class ResNet_loco(nn.Module):
  """ResNet model."""

  def __init__(
      self,
      blocks_per_group: Sequence[int],
      channels_per_group: Sequence[int] = (64, 128, 256, 512),
      use_projection: Sequence[bool] = (True, True, True, True),
      strides: Sequence[int] = (1, 2, 2, 2),
  ):
    """Initializes a ResNet model with customizable layers and configurations.

    This constructor allows defining the architecture of a ResNet model by
    setting the number of blocks, channels, projection usage, and strides for
    each group of blocks within the network. It provides flexibility in
    creating various ResNet configurations.

    Args:
      blocks_per_group: A sequence of 4 integers, each indicating the number
        of residual blocks in each group.
      channels_per_group: A sequence of 4 integers, each specifying the number
        of output channels for the blocks in each group. Defaults to (64, 128,
        256, 512).
      use_projection: A sequence of 4 booleans, each indicating whether to use
        a projection shortcut (True) or an identity shortcut (False) in each
        group. Defaults to (True, True, True, True).
      strides: A sequence of 4 integers, each specifying the stride size for
        the convolutions in each group. Defaults to (1, 2, 2, 2).

    The ResNet model created will have 4 groups, with each group's
    architecture defined by the corresponding elements in these sequences.
    """
    super().__init__()

    self.initial_conv = Conv2dSamePadding(
        in_channels=3,
        out_channels=channels_per_group[0],
        kernel_size=(7, 7),
        stride=2,
        padding=0,
        bias=False,
    )

    block_groups = []
    for i, _ in enumerate(strides):
      block_groups.append(
          BlockGroup_loco(
              channels_in=channels_per_group[i - 1] if i > 0 else 64,
              channels_out=channels_per_group[i],
              num_blocks=blocks_per_group[i],
              stride=strides[i],
              use_projection=use_projection[i],
          )
      )
    self.block_groups = nn.ModuleList(block_groups)

  def forward(self, inputs):
    result = {}
    out = inputs
    out = self.initial_conv(out)
    result['initial_conv'] = out

    for block_id, block_group in enumerate(self.block_groups):
      out = block_group(out)
      result[f'resnet_unit_{block_id}'] = out

    return result


class Conv2dSamePadding(torch.nn.Conv2d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
      return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      ih, iw = x.size()[-2:]

      pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
      pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])
  
      if pad_h > 0 or pad_w > 0:
        x = F.pad(
            x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
        )
      return F.conv2d(
        x,
        self.weight,
        self.bias,
        self.stride,
        # self.padding,
        0,
        self.dilation,
        self.groups,
      )

class CMDTop(nn.Module):
    def __init__(self, in_channel, out_channels, kernel_shapes, strides):
        super(CMDTop, self).__init__()
        self.in_channels = [in_channel] + list(out_channels[:-1])
        self.out_channels = out_channels
        self.kernel_shapes = kernel_shapes
        self.strides = strides

        self.conv = nn.ModuleList([
            nn.Sequential(
                Conv2dSamePadding(
                    in_channels=self.in_channels[i],
                    out_channels=self.out_channels[i],
                    kernel_size=self.kernel_shapes[i],
                    stride=self.strides[i],
                ),
                nn.GroupNorm(out_channels[i] // 16, out_channels[i]),
                nn.ReLU()
            ) for i in range(len(out_channels))
        ])

    def forward(self, x):
        """
        x: (b, h, w, i, j)
        """
        out1 = rearrange(x, 'b h w i j -> b (i j) h w')
        out2 = rearrange(x, 'b h w i j -> b (h w) i j')
        
        for i in range(len(self.out_channels)):
            out1 = self.conv[i](out1)
        
        for i in range(len(self.out_channels)):
            out2 = self.conv[i](out2)

        out1 = torch.mean(out1, dim=(2, 3)) # (b, out_channels[-1])
        out2 = torch.mean(out2, dim=(2, 3)) # (b, out_channels[-1])

        return torch.cat([out1, out2], dim=-1) # (b, 2*out_channels[-1])


class BlockV2(nn.Module):
  """ResNet V2 block."""

  def __init__(
      self,
      channels_in: int,
      channels_out: int,
      stride: Union[int, Sequence[int]],
      use_projection: bool,
  ):
    super().__init__()
    self.padding = (1, 1, 1, 1)
    # Handle assymetric padding created by padding="SAME" in JAX/LAX
    if stride == 1:
      self.padding = (1, 1, 1, 1)
    elif stride == 2:
      self.padding = (0, 2, 0, 2)
    else:
      raise ValueError(
          'Check correct padding using padtype_to_padsin jax._src.lax.lax'
      )

    self.use_projection = use_projection
    if self.use_projection:
      self.proj_conv = nn.Conv2d(
          in_channels=channels_in,
          out_channels=channels_out,
          kernel_size=1,
          stride=stride,
          padding=0,
          bias=False,
      )

    self.bn_0 = nn.InstanceNorm2d(
        num_features=channels_in,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=False,
    )
    self.conv_0 = nn.Conv2d(
        in_channels=channels_in,
        out_channels=channels_out,
        kernel_size=3,
        stride=stride,
        padding=0,
        bias=False,
    )

    self.conv_1 = nn.Conv2d(
        in_channels=channels_out,
        out_channels=channels_out,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    self.bn_1 = nn.InstanceNorm2d(
        num_features=channels_out,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=False,
    )

  def forward(self, inputs):
    x = shortcut = inputs

    x = self.bn_0(x)
    x = torch.relu(x)
    if self.use_projection:
      shortcut = self.proj_conv(x)

    x = self.conv_0(F.pad(x, self.padding))

    x = self.bn_1(x)
    x = torch.relu(x)
    # no issues with padding here as this layer always has stride 1
    x = self.conv_1(x)

    return x + shortcut


class BlockGroup(nn.Module):
  """Higher level block for ResNet implementation."""

  def __init__(
      self,
      channels_in: int,
      channels_out: int,
      num_blocks: int,
      stride: Union[int, Sequence[int]],
      use_projection: bool,
  ):
    super().__init__()
    blocks = []
    for i in range(num_blocks):
      blocks.append(
          BlockV2(
              channels_in=channels_in if i == 0 else channels_out,
              channels_out=channels_out,
              stride=(1 if i else stride),
              use_projection=(i == 0 and use_projection),
          )
      )
    self.blocks = nn.ModuleList(blocks)

  def forward(self, inputs):
    out = inputs
    for block in self.blocks:
      out = block(out)
    return out
  

class ResNet(nn.Module):
  """ResNet model."""

  def __init__(
      self,
      blocks_per_group: Sequence[int],
      channels_per_group: Sequence[int] = (64, 128, 256, 512),
      use_projection: Sequence[bool] = (True, True, True, True),
      strides: Sequence[int] = (1, 2, 2, 2),
  ):
    """Initializes a ResNet model with customizable layers and configurations.

    This constructor allows defining the architecture of a ResNet model by
    setting the number of blocks, channels, projection usage, and strides for
    each group of blocks within the network. It provides flexibility in
    creating various ResNet configurations.

    Args:
      blocks_per_group: A sequence of 4 integers, each indicating the number
        of residual blocks in each group.
      channels_per_group: A sequence of 4 integers, each specifying the number
        of output channels for the blocks in each group. Defaults to (64, 128,
        256, 512).
      use_projection: A sequence of 4 booleans, each indicating whether to use
        a projection shortcut (True) or an identity shortcut (False) in each
        group. Defaults to (True, True, True, True).
      strides: A sequence of 4 integers, each specifying the stride size for
        the convolutions in each group. Defaults to (1, 2, 2, 2).

    The ResNet model created will have 4 groups, with each group's
    architecture defined by the corresponding elements in these sequences.
    """
    super().__init__()

    self.initial_conv = nn.Conv2d(
        in_channels=4,
        out_channels=channels_per_group[0],
        kernel_size=(7, 7),
        stride=2,
        padding=0,
        bias=False,
    )

    block_groups = []
    for i, _ in enumerate(strides):
      block_groups.append(
          BlockGroup(
              channels_in=channels_per_group[i - 1] if i > 0 else 32,#64,   Azad
              channels_out=channels_per_group[i],
              num_blocks=blocks_per_group[i],
              stride=strides[i],
              use_projection=use_projection[i],
          )
      )
    self.block_groups = nn.ModuleList(block_groups)
    self.downsampler = nn.PixelUnshuffle(2)
    # print(channels_per_group[-1], int(channels_per_group[-1]/2))
    # raise KeyboardInterrupt
    #self.upsampler = CARAFE(in_channels=channels_per_group[-1], out_channels=int(channels_per_group[-1]/2), scale=2)
    #self.upsampler = nn.PixelShuffle(2)
  def forward(self, inputs):
    result = {}
    out = inputs
    # frame_list = torch.linspace(0, 15, steps=7).long()
    # chnl = 30
    # from utils.utils_ import visualize_feature_maps, fmaps2vid
    # visualize_feature_maps(
    #     inputs.permute(1,0, 2, 3), b=0, c_list=frame_list, 
    #     name="Frame", channel=f"/channel {chnl}", cmap='gray'
    # ) # 1st channel from all frames


    # print(inputs.shape)
    out = self.downsampler(out)
    # print(f"Hello, {out.shape}")
    # visualize_feature_maps(
    #     out.permute(1,0, 2, 3), b=3, c_list=frame_list, 
    #     name="Frame", channel=f"/channel {chnl}", cmap='gray', save_path="heat_maps.png"
    # ) # 1st channel from all frames
    # raise KeyboardInterrupt
    out = self.initial_conv(F.pad(out, (2, 4, 2, 4)))
    result['initial_conv'] = out

    for block_id, block_group in enumerate(self.block_groups):
      out = block_group(out)
      result[f'resnet_unit_{block_id}'] = out
    #   print(out.shape)
    # raise KeyboardInterrupt
    
    # out = self.upsampler(result[f'resnet_unit_3'])
    # print(out.shape, result[f'resnet_unit_3'].shape)
    # raise KeyboardInterrupt
    # result[f'resnet_unit_3'] = out
    # print(result[f'resnet_unit_3'].shape)
    # raise KeyboardInterrupt
    return result


class ExtraConvBlock(nn.Module):
  """Additional convolution block."""

  def __init__(
      self,
      channel_dim,
      channel_multiplier,
  ):
    super().__init__()
    self.channel_dim = channel_dim
    self.channel_multiplier = channel_multiplier

    self.layer_norm = nn.LayerNorm(
        normalized_shape=channel_dim, elementwise_affine=True, bias=True
    )
    self.conv = nn.Conv2d(
        self.channel_dim * 3,
        self.channel_dim * self.channel_multiplier,
        kernel_size=3,
        stride=1,
        padding=1,
    )
    self.conv_1 = nn.Conv2d(
        self.channel_dim * self.channel_multiplier,
        self.channel_dim,
        kernel_size=3,
        stride=1,
        padding=1,
    )

  def forward(self, x):
    x = self.layer_norm(x)
    x = x.permute(0, 3, 1, 2)
    prev_frame = torch.cat([x[0:1], x[:-1]], dim=0)
    next_frame = torch.cat([x[1:], x[-1:]], dim=0)
    resid = torch.cat([x, prev_frame, next_frame], axis=1)
    resid = self.conv(resid)
    resid = F.gelu(resid, approximate='tanh')
    x += self.conv_1(resid)
    x = x.permute(0, 2, 3, 1)
    return x


class ExtraConvs(nn.Module):
  """Additional CNN."""

  def __init__(
      self,
      num_layers=5,
      channel_dim=256,
      channel_multiplier=4,
  ):
    super().__init__()
    self.num_layers = num_layers
    self.channel_dim = channel_dim
    self.channel_multiplier = channel_multiplier

    self.blocks = nn.ModuleList()
    for _ in range(self.num_layers):
      self.blocks.append(
          ExtraConvBlock(self.channel_dim, self.channel_multiplier)
      )
    #self.upsampler = CARAFE(in_channels=channel_dim, out_channels=int(channel_dim/2), scale=2)
    

  def forward(self, x):
    for block in self.blocks:
      x = block(x)
    #x = self.upsampler(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    return x

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

  lowres: Sequence[torch.Tensor]
  hires: Sequence[torch.Tensor]
  feats: Sequence[torch.Tensor]
  resolutions: Sequence[Tuple[int, int]]


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

  lowres: Sequence[torch.Tensor]
  hires: Sequence[torch.Tensor]
  feats: Sequence[torch.Tensor]
  resolutions: Sequence[Tuple[int, int]]

class BasicEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=128, stride=8, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.stride = stride
        self.norm_fn = norm_fn

        self.in_planes = 32
        
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=self.in_planes)
            self.norm2 = nn.GroupNorm(num_groups=8, num_channels=output_dim*2)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.InstanceNorm2d(self.in_planes)
            self.norm2 = nn.InstanceNorm2d(output_dim*2)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(self.in_planes)
            self.norm2 = nn.InstanceNorm2d(output_dim*2)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()
            
        self.conv1 = nn.Conv2d(input_dim, self.in_planes, kernel_size=7, stride=2, padding=3, padding_mode='zeros')
        self.relu1 = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(32,  stride=1)
        self.layer2 = self._make_layer(48, stride=2)
        self.layer3 = self._make_layer(64, stride=2)
        self.layer4 = self._make_layer(64, stride=2)

        self.conv2 = nn.Conv2d(64+64+48+32, output_dim*2, kernel_size=3, padding=1, padding_mode='zeros')
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_dim*2, output_dim, kernel_size=1)
        
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.InstanceNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock2d(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock2d(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):

        _, _, H, W = x.shape
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        a = self.layer1(x)
        b = self.layer2(a)
        c = self.layer3(b)
        d = self.layer4(c)
        a = F.interpolate(a, (H//self.stride, W//self.stride), mode='bilinear', align_corners=True)
        b = F.interpolate(b, (H//self.stride, W//self.stride), mode='bilinear', align_corners=True)
        c = F.interpolate(c, (H//self.stride, W//self.stride), mode='bilinear', align_corners=True)
        d = F.interpolate(d, (H//self.stride, W//self.stride), mode='bilinear', align_corners=True)
        x = self.conv2(torch.cat([a,b,c,d], dim=1))
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)
        return x


class Conv1dPad(nn.Module):
    """
    nn.Conv1d with auto-computed padding ("same" padding)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(Conv1dPad, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        net = x
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0.0)
        net = self.conv(net)
        return net


class ResidualBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, use_norm, use_do, is_first_block=False):
        super(ResidualBlock1d, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.stride = 1
        self.is_first_block = is_first_block
        self.use_norm = use_norm
        self.use_do = use_do

        self.norm1 = nn.InstanceNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = Conv1dPad(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride,
            groups=self.groups)

        self.norm2 = nn.InstanceNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = Conv1dPad(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1,
            groups=self.groups)

    def forward(self, x):
        
        identity = x
        
        out = x
        if not self.is_first_block:
            if self.use_norm:
                out = self.norm1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        
        if self.use_norm:
            out = self.norm2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
            
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0.0)
            identity = identity.transpose(-1,-2)
        
        out += identity
        return out

class ResidualBlock2d(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock2d, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride, padding_mode='zeros')
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, padding_mode='zeros')
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd), indexing='ij')
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


class CorrBlock:
    def __init__(self, fmaps: torch.Tensor, num_levels: int = 4, radius: int = 4):
        #super(CorrBlock, self).__init__()
        B, S, C, H, W = fmaps.shape
        self.S = S
        self.C = C
        self.H = H
        self.W = W
        self.num_levels = num_levels
        self.radius = radius

        # Initialize feature maps pyramid as a list of tensors
        self.fmaps_pyramid: List[torch.Tensor] = []
        self.fmaps_pyramid.append(fmaps)
        self.corrs_pyramid: List[torch.Tensor] = []

        # Construct the feature maps pyramid using average pooling
        for i in range(self.num_levels - 1):
            fmaps_ = fmaps.reshape(B * S, C, H, W)
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            _, _, H, W = fmaps_.shape
            fmaps = fmaps_.reshape(B, S, C, H, W)
            self.fmaps_pyramid.append(fmaps)
        

    def sample(self, coords: torch.Tensor, chunk_size: int =64)-> torch.Tensor:
        r = self.radius
        B, S, N, D = coords.shape
        assert D == 2

        # x0 = coords[:, 0, :, 0].round().clamp(0, self.W - 1).long()
        # y0 = coords[:, 0, :, 1].round().clamp(0, self.H - 1).long()

        #out_pyramid = []
        out_pyramid: List[torch.Tensor] = []

        # Process in chunks
        for i in range(self.num_levels):
            corrs = self.corrs_pyramid[i]  # B, S, N, H, W
            _, _, _, H, W = corrs.shape

            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), dim=-1).to(coords.device)

            if chunk_size > 0:
                coords_lvl_list: List[torch.Tensor] = []
                for chunk_start in range(0, B * S * N, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, B * S * N)
                    centroid_lvl = coords.reshape(B * S * N, 1, 1, 2)[chunk_start:chunk_end] / 2**i
                    delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
                    coords_lvl = centroid_lvl + delta_lvl
                    coords_lvl_list.append(coords_lvl)
                coords_lvl = torch.cat(coords_lvl_list, dim=0)
            else:
                centroid_lvl = coords.reshape(B*S*N, 1, 1, 2) / 2**i
                delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
                coords_lvl = centroid_lvl + delta_lvl

            
            if chunk_size > 0:
                # Process correlation for the chunked coordinates
                corrs_list: List[torch.Tensor] = []
                for chunk_start in range(0, B * S * N, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, B * S * N)
                    corrs_chunk = bilinear_sampler(
                        corrs.reshape(B * S * N, 1, H, W)[chunk_start:chunk_end], 
                        coords_lvl[chunk_start:chunk_end]
                    )
                    corrs_list.append(corrs_chunk.view(-1, 2 * r + 1, 2 * r + 1))                
                corrs = torch.cat(corrs_list, dim=0)

            else:
                corrs = bilinear_sampler(corrs.reshape(B*S*N, 1, H, W), coords_lvl)
                #corrs = corrs.view(B, S, N, -1)
            
            out_pyramid.append(corrs.view(B, S, N, -1))

        out = torch.cat(out_pyramid, dim=-1)  # B, S, N, LRR*2
        return out.contiguous().float()


    def corr(self, targets: torch.Tensor, chunk_size: int=0):
        B, S, N, C = targets.shape
        assert C == self.C

        self.corrs_pyramid = []

        for fmaps in self.fmaps_pyramid:
            _, _, _, H, W = fmaps.shape
            fmap2s = fmaps.view(B, S, C, H * W)
            
            if chunk_size > 0:
                corrs_list: List[torch.Tensor] = []
                for chunk_start in range(0, N, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, N)
                    corrs_chunk = torch.matmul(
                        targets[:, :, chunk_start:chunk_end, :], fmap2s
                    )
                    corrs_chunk = corrs_chunk / torch.sqrt(torch.tensor(C).float())
                    corrs_list.append(corrs_chunk)

                corrs = torch.cat(corrs_list, dim=2)
            else:
                corrs = torch.matmul(targets, fmap2s)
                corrs = corrs / torch.sqrt(torch.tensor(C).float())

            corrs = corrs.view(B, S, N, H, W)
            self.corrs_pyramid.append(corrs)


class DeltaBlock(nn.Module):
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 128, corr_levels: int = 4, corr_radius: int = 3):
        super(DeltaBlock, self).__init__()
        
        kitchen_dim = (corr_levels * (2 * corr_radius + 1) ** 2) + latent_dim + 2 + 50 * 60 + 2

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        in_channels = kitchen_dim
        base_filters = 64
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
                is_first_block=is_first_block
            )
            self.basicblock_list.append(tmp_block)

        self.final_norm = nn.InstanceNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        self.dense = nn.Linear(out_channels, 2)
        
    def forward(self, fcorr: torch.Tensor, flow: torch.Tensor, fflow: torch.Tensor, xys0: torch.Tensor) -> torch.Tensor:
        B, S, D = flow.shape
        assert D == 2
        flow_sincos = posemb_sincos_2d_xy(flow, self.latent_dim, cat_coords=True)

        x = torch.cat([fcorr, fflow, flow_sincos, xys0], dim=2)  # B, S, -1
        out = x.permute(0, 2, 1)
        out = self.first_block_conv(out)
        out = self.first_block_relu(out)
        # for i_block in range(self.n_block):
        #     net = self.basicblock_list[i_block]
        #     out = net(out)
        for i, block in enumerate(self.basicblock_list):
            out = block(out)
        out = self.final_relu(out)
        out = out.permute(0, 2, 1)
        
        delta = self.dense(out)
        return delta    
    


class DeltaBlock_temp(nn.Module):
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 128, corr_levels: int = 4, corr_radius: int = 3):
        super(DeltaBlock_temp, self).__init__()

        kitchen_dim = (corr_levels * (2 * corr_radius + 1) ** 2) + latent_dim + 2 + 50 * 60 + 2 + 196

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        in_channels = kitchen_dim
        base_filters = 64
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
                is_first_block=is_first_block
            )
            self.basicblock_list.append(tmp_block)

        self.final_norm = nn.InstanceNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        self.dense = nn.Linear(out_channels, 2)

    def forward(self, fcorr: torch.Tensor, flow: torch.Tensor, fflow: torch.Tensor, xys0: torch.Tensor) -> torch.Tensor:
        B, S, D = flow.shape
        assert D == 2
        flow_sincos = posemb_sincos_2d_xy(flow, self.latent_dim, cat_coords=True)
        
        x = torch.cat([fcorr, fflow, flow_sincos, xys0], dim=2)  # B, S, -1
        out = x.permute(0, 2, 1)
        out = self.first_block_conv(out)
        out = self.first_block_relu(out)
        # for i_block in range(self.n_block):
        #     net = self.basicblock_list[i_block]
        #     out = net(out)
        for i, block in enumerate(self.basicblock_list):
            out = block(out)
        out = self.final_relu(out)
        out = out.permute(0, 2, 1)
        
        delta = self.dense(out)
        return delta    
