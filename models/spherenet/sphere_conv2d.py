import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math

from .grid_generator import (
    GridGenerator, 
    GridSampler,
    IncreIntervalGridGenerator,
    GridSamplerNewTextureNoGrad,
    GridGeneratorPatchCoordsFixBorder
    )


class SphereConv2d(nn.Conv2d):
  """
  kernel_size: (H, W)
  """

  def __init__(self, in_channels: int, out_channels: int, kernel_size=(3, 3),
               stride=1, padding=0, dilation=1, scale=None,
               groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'):
    super(SphereConv2d, self).__init__(
      in_channels, out_channels, kernel_size,
      stride, padding, dilation, groups, bias, padding_mode)
    self.grid_shape = None
    self.grid = None
    self.scale = scale
    self.sampler = GridSampler()

  def genSamplingPattern(self, h, w):
    with torch.no_grad():
      gridGenerator = GridGenerator(h, w, self.kernel_size, self.stride)
      LonLatSamplingPattern = gridGenerator.createSamplingPattern()

      # generate grid to use `F.grid_sample`
      lat_grid = (LonLatSamplingPattern[:, :, :, 0] / h) * 2 - 1
      lon_grid = (LonLatSamplingPattern[:, :, :, 1] / w) * 2 - 1

      grid = np.stack((lon_grid, lat_grid), axis=-1)
    with torch.no_grad():
      self.grid = torch.FloatTensor(grid)
      self.grid.requires_grad = False

  def forward(self, x):
    # Generate Sampling Pattern
    B, C, H, W = x.shape
    with torch.no_grad():
      if (self.grid_shape is None) or (self.grid_shape != (H, W)):
        self.grid_shape = (H, W)
        self.genSamplingPattern(H, W)

    with torch.no_grad():
      grid = self.grid.repeat((B, 1, 1, 1)).to(x.device)  # (B, H*Kh, W*Kw, 2)
      grid.requires_grad = False
    
    x = self.sampler(x, grid)
    # x = F.grid_sample(x, grid, align_corners=True, mode='nearest')  # (B, in_c, H*Kh, W*Kw)

    # self.weight -> (out_c, in_c, Kh, Kw) 
    if self.scale:
      x = F.conv2d(x, self.weight * self.scale, self.bias, stride=self.kernel_size, padding=self.padding)
    else:
      x = F.conv2d(x, self.weight, self.bias, stride=self.kernel_size, padding=self.padding)

    return x  # (B, out_c, H/stride_h, W/stride_w)


class IncreIntervalSphereConv2d(nn.Conv2d):
  """
  kernel_size: (H, W)
  """

  def __init__(self, in_channels: int, out_channels: int, kernel_size=(3, 3),
               stride=1, padding=0, dilation=1, scale=None,
               groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'):
    super(IncreIntervalSphereConv2d, self).__init__(
      in_channels, out_channels, kernel_size,
      stride, padding, dilation, groups, bias, padding_mode)
    self.grid_shape = None
    self.grid = None
    self.scale = scale
    self.sampler = GridSampler()

  def genSamplingPattern(self, h, w):
    with torch.no_grad():
      gridGenerator = IncreIntervalGridGenerator(h, w, self.kernel_size, self.stride)
      LonLatSamplingPattern = gridGenerator.createSamplingPattern()

      # generate grid to use `F.grid_sample`
      lat_grid = (LonLatSamplingPattern[:, :, :, 0] / h) * 2 - 1
      lon_grid = (LonLatSamplingPattern[:, :, :, 1] / w) * 2 - 1

      grid = np.stack((lon_grid, lat_grid), axis=-1)
    with torch.no_grad():
      self.grid = torch.FloatTensor(grid)
      self.grid.requires_grad = False

  def forward(self, x):
    # Generate Sampling Pattern
    B, C, H, W = x.shape
    with torch.no_grad():
      if (self.grid_shape is None) or (self.grid_shape != (H, W)):
        self.grid_shape = (H, W)
        self.genSamplingPattern(H, W)

    with torch.no_grad():
      grid = self.grid.repeat((B, 1, 1, 1)).to(x.device)  # (B, H*Kh, W*Kw, 2)
      grid.requires_grad = False
    
    x = self.sampler(x, grid)
    # x = F.grid_sample(x, grid, align_corners=True, mode='nearest')  # (B, in_c, H*Kh, W*Kw)

    # self.weight -> (out_c, in_c, Kh, Kw) 
    if self.scale:
      x = F.conv2d(x, self.weight * self.scale, self.bias, stride=self.kernel_size, padding=self.padding)
    else:
      x = F.conv2d(x, self.weight, self.bias, stride=self.kernel_size, padding=self.padding)

    return x  # (B, out_c, H/stride_h, W/stride_w)


class SphereConvBatchDiffFixBorderGNoGrad(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=(3, 3),
               stride=1, padding=0, dilation=1,
               groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'):
        super(SphereConvBatchDiffFixBorderGNoGrad, self).__init__(
        in_channels, out_channels, kernel_size,
        stride, padding, dilation, groups, bias, padding_mode)
        self.grid_shape = None
        self.grid = None
        self.scale = 1 / math.sqrt(in_channels * kernel_size[0] ** 2)
        # spectural norm
        self.sampler = GridSamplerNewTextureNoGrad()
        self.activation = nn.LeakyReLU()
        self.weight = nn.Parameter(
                torch.Tensor(
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]
                ]
                ).repeat(out_channels, in_channels, 1, 1)
            )

    def genSamplingPattern(self, h, w, stride, coords_partial):
        with torch.no_grad():
            gridGenerator = GridGeneratorPatchCoordsFixBorder(h, w, self.kernel_size, stride=stride, coords_partial=coords_partial)
            LonLatSamplingPattern = gridGenerator.createSamplingPattern()

            # generate grid to use `F.grid_sample`
            lat_grid = (LonLatSamplingPattern[:, :, :, 0] / coords_partial["x_total"]) * 2 - 1
            lon_grid = (LonLatSamplingPattern[:, :, :, 1] / coords_partial["y_total"]) * 2 - 1
            
            grid = np.stack((lon_grid, lat_grid), axis=-1)
            if not coords_partial.get("test_flag", False):
                with torch.no_grad():
                    grid = torch.FloatTensor(grid)
                    grid.requires_grad = False
                    return grid
            else:
                with torch.no_grad():
                    self.grid = torch.FloatTensor(grid)
                    self.grid.requires_grad = False

    def forward(self, x, coords_partial):
        # Generate Sampling Pattern
        B, C, H, W = x.shape
        with torch.no_grad():
            if isinstance(coords_partial, list):
                j_test = False
            else:
                j_test = coords_partial.get("test_flag", False)

            if not j_test:
                assert isinstance(coords_partial, list), "There is an error in coordhandler for training"
                self.grid_shape = (H, W)
                grids = None
                for img_coords_partial in coords_partial:
                    grid = self.genSamplingPattern(H, W, stride=1, coords_partial=img_coords_partial)
                    if grids is None:
                        grids = grid
                    else:
                        grids = torch.cat((grids, grid), dim=0)
                
                grid = grids.to(x.device)
                grid.requires_grad = False
                    
            else:
                self.grid_shape = (H, W)
                self.genSamplingPattern(H, W, stride=1, coords_partial=coords_partial)
                grid = self.grid.repeat((B, 1, 1, 1)).to(x.device)  # (B, H*Kh, W*Kw, 2)
                grid.requires_grad = False
        
        x = self.sampler(x, grid)
        # (B, in_c, H*Kh, W*Kw)

        # self.weight -> (out_c, in_c, Kh, Kw) 
        if self.scale:
            x = F.conv2d(x, self.weight * self.scale, self.bias, stride=self.kernel_size, padding=self.padding)
        else:
            x = F.conv2d(x, self.weight, self.bias, stride=self.kernel_size, padding=self.padding)

        return self.activation(x)  # (B, out_c, H/stride_h, W/stride_w)

