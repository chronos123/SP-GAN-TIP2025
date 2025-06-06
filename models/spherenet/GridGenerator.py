import numpy as np
import torch
from functools import lru_cache
from torch import nn
import torch.nn.functional as F
import logging

from .grid_sample_grad_fix import grid_sample
from .grid_sample_ops import grid_sample_github


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(filename)s %(levelname)s %(message)s',
    datefmt='%a %d %b %Y %H:%M:%S',
    filename='logs/check.log',
    filemode='a+'
)


__logger__ = logging.getLogger(__name__)


class GridGenerator:
    def __init__(self, height: int, width: int, kernel_size, stride=1):
        if isinstance(height, torch.Tensor):
            height = height.numpy()
            width = width.numpy()
        self.height = height
        self.width = width
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)  # (Kh, Kw)
        else:
            self.kernel_size = kernel_size
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:  
            self.stride = stride  # (H, W)

    def createSamplingPattern(self):
        """
    :return: (1, H*Kh, W*Kw, (Lat, Lon)) sampling pattern
    """
        kerX, kerY = self.createKernel()  # (Kh, Kw)

        # create some values using in generating lat/lon sampling pattern
        rho = np.sqrt(kerX**2 + kerY**2)

        Kh, Kw = self.kernel_size
        # when the value of rho at center is zero, some lat values explode to `nan`.
        if Kh % 2 and Kw % 2:
            rho[Kh // 2][Kw // 2] = 1e-8

        nu = np.arctan(rho)
        cos_nu = np.cos(nu)
        sin_nu = np.sin(nu)

        stride_h, stride_w = self.stride
        h_range = np.arange(0, self.height, stride_h)
        w_range = np.arange(0, self.width, stride_w)

        lat_range = ((h_range / self.height) - 0.5) * np.pi
        lon_range = ((w_range / self.width) - 0.5) * (2 * np.pi)

        # generate latitude sampling pattern
        lat = np.array([
            np.arcsin(cos_nu * np.sin(_lat) +
                      kerY * sin_nu * np.cos(_lat) / rho) for _lat in lat_range
        ])  # (H, Kh, Kw)

        lat = np.array([lat for _ in lon_range])  # (W, H, Kh, Kw)
        lat = lat.transpose((1, 0, 2, 3))  # (H, W, Kh, Kw)

        # generate longitude sampling pattern
        lon = np.array([
            np.arctan(
                kerX * sin_nu /
                (rho * np.cos(_lat) * cos_nu - kerY * np.sin(_lat) * sin_nu))
            for _lat in lat_range
        ])  # (H, Kh, Kw)

        lon = np.array([lon + _lon for _lon in lon_range])  # (W, H, Kh, Kw)
        lon = lon.transpose((1, 0, 2, 3))  # (H, W, Kh, Kw)

        # (radian) -> (index of pixel)
        lat = (lat / np.pi + 0.5) * self.height
        lon = ((lon / (2 * np.pi) + 0.5) * self.width) % self.width

        LatLon = np.stack(
            (lat, lon))  # (2, H, W, Kh, Kw) = ((lat, lon), H, W, Kh, Kw)
        LatLon = LatLon.transpose(
            (1, 3, 2, 4, 0))  # (H, Kh, W, Kw, 2) = (H, Kh, W, Kw, (lat, lon))

        H, Kh, W, Kw, d = LatLon.shape
        LatLon = LatLon.reshape((1, H * Kh, W * Kw, d))  # (1, H*Kh, W*Kw, 2)

        return LatLon

    def createKernel(self):
        """
    :return: (Ky, Kx) kernel pattern
    """

        Kh, Kw = self.kernel_size

        delta_lat = np.pi / self.height
        delta_lon = 2 * np.pi / self.width

        range_x = np.arange(-(Kw // 2), Kw // 2 + 1)
        if not Kw % 2:
            range_x = np.delete(range_x, Kw // 2)

        range_y = np.arange(-(Kh // 2), Kh // 2 + 1)
        if not Kh % 2:
            range_y = np.delete(range_y, Kh // 2)

        kerX = np.tan(range_x * delta_lon)
        kerY = np.tan(range_y * delta_lat) / np.cos(range_y * delta_lon)

        return np.meshgrid(kerX, kerY)  # (Kh, Kw)


class GridGeneratorPatchCoordsFixBorder:
    global_id = 0
    layer_id = 0
    coords_diff = {
        1: None,
        2: None,
        3: None,
        4: None
    }
    def __init__(self, height: int, width: int, kernel_size, stride=1, coords_partial=None,):
        if isinstance(height, torch.Tensor):
            height = height.numpy()
            width = width.numpy()
        self.height = height
        self.width = width
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)  # (Kh, Kw)
        else:
            self.kernel_size = kernel_size
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:  
            self.stride = stride  # (H, W)
        self.coords_partial = coords_partial
        assert self.coords_partial is not None

    @lru_cache
    def createSamplingPattern(self):
        """
    :return: (1, H*Kh, W*Kw, (Lat, Lon)) sampling pattern
    """

        kerX, kerY = self.createKernel()  # (Kh, Kw)

        # create some values using in generating lat/lon sampling pattern
        rho = np.sqrt(kerX**2 + kerY**2)

        Kh, Kw = self.kernel_size
        # when the value of rho at center is zero, some lat values explode to `nan`.
        if Kh % 2 and Kw % 2:
            rho[Kh // 2][Kw // 2] = 1e-8

        nu = np.arctan(rho)
        cos_nu = np.cos(nu)
        sin_nu = np.sin(nu)

        stride_h, stride_w = self.stride
        h_range = np.arange(0, self.height, stride_h)
        w_range = np.arange(0, self.width, stride_w)
        
        out_h = len(h_range)
        out_w = len(w_range)
        
        partial = 0.8

        if self.coords_partial.get("test_flag", False):
            partial =  self.coords_partial.get("partial", partial)

        if self.coords_partial.get("test_flag", False) and self.coords_partial.get("full_shape", None):
            assert out_w == 35 and out_h ==35

            height, width = self.coords_partial["full_shape"]
            
            x_st_idx = round(self.coords_partial["p_x_st"] * self.coords_partial['x_total'])
            x_ed_idx = round(self.coords_partial["p_x_ed"] * self.coords_partial['x_total']) - 1
            y_st_idx = round(self.coords_partial["p_y_st"] * self.coords_partial['y_total'])
            y_ed_idx = round(self.coords_partial["p_y_ed"] * self.coords_partial['y_total']) - 1

            all_x_centers = np.linspace(-np.pi * partial/2, np.pi * partial/2, height)
            all_y_centers = np.linspace(-np.pi, np.pi, width)

            if self.coords_partial["circular_flag"]:
                y_ed_idx = y_ed_idx % width
                lat_range = all_x_centers[x_st_idx:x_ed_idx]
                lon_range_st = all_y_centers[y_st_idx:]
                lon_range_ed = all_y_centers[:y_ed_idx] + np.pi * 2
                lon_range = np.concatenate((lon_range_st, lon_range_ed), axis=0)

            else:
                lat_range = all_x_centers[x_st_idx:x_ed_idx]
                lon_range = all_y_centers[y_st_idx:y_ed_idx]
        
        elif self.coords_partial.get("full_shape", None) and self.coords_partial.get("pre_sample_mode", None):
            assert out_w == 35 and out_h ==35

            height, width = self.coords_partial["full_shape"]
            
            x_st_idx = round(self.coords_partial["p_x_st"] * self.coords_partial['x_total'])
            x_ed_idx = round(self.coords_partial["p_x_ed"] * self.coords_partial['x_total']) + 1
            y_st_idx = round(self.coords_partial["p_y_st"] * self.coords_partial['y_total'])
            y_ed_idx = round(self.coords_partial["p_y_ed"] * self.coords_partial['y_total']) + 1

            all_x_centers = np.linspace(-np.pi * partial/2, np.pi * partial/2, height)
            all_y_centers = np.linspace(-np.pi, np.pi, width)

            if self.coords_partial["circular_flag"]:
                if y_ed_idx == width:
                    lat_range = all_x_centers[x_st_idx:x_ed_idx]
                    lon_range = all_y_centers[y_st_idx:y_ed_idx]
                else:
                    y_ed_idx = y_ed_idx % width
                    lat_range = all_x_centers[x_st_idx:x_ed_idx]
                    lon_range_st = all_y_centers[y_st_idx:]
                    lon_range_ed = all_y_centers[:y_ed_idx] + np.pi * 2
                    lon_range = np.concatenate((lon_range_st, lon_range_ed), axis=0)

            else:
                lat_range = all_x_centers[x_st_idx:x_ed_idx]
                lon_range = all_y_centers[y_st_idx:y_ed_idx]
            assert len(lon_range) == 35, f"sample error: {lon_range}\n len is {len(lon_range)}\n from {y_st_idx} to {y_ed_idx}"\
            f"\tst: {self.coords_partial['y_st']}, ed:{self.coords_partial['y_ed']}, width:{width}"
        else:
            x_st =  self.coords_partial["p_x_st"] * np.pi * partial
            x_ed = self.coords_partial["p_x_ed"] * np.pi * partial
            y_st = self.coords_partial["p_y_st"] * np.pi * 2
            y_ed = (self.coords_partial["p_y_ed"] * np.pi * 2)

            if y_ed == 2 * np.pi:
                pass
            else:
                y_ed = y_ed % (np.pi * 2)

            x_st = self.convert_numpy(x_st)
            x_ed = self.convert_numpy(x_ed)
            y_st = self.convert_numpy(y_st)
            y_ed = self.convert_numpy(y_ed)
            
            if self.coords_partial["circular_flag"]:
                lat_range = np.linspace(x_st, x_ed, out_h) - (np.pi / 2 * partial)

                y_ed = y_ed + 2 * np.pi
                lon_range = np.linspace(y_st, y_ed, out_w) - np.pi
                
            else:
                lat_range = np.linspace(x_st, x_ed, out_h) - (np.pi / 2 * partial)
                lon_range = np.linspace(y_st, y_ed, out_w) - np.pi

        # generate latitude sampling pattern
        lat = np.array([
            np.arcsin(cos_nu * np.sin(_lat) +
                    kerY * sin_nu * np.cos(_lat) / rho) for _lat in lat_range
        ])  # (H, Kh, Kw)

        lat_range_norm = self.min_max_norm(lat_range)
        lat_pattern = self.get_pattern(lat, (Kh, Kw)) # (35, 3, 3)
        lat_norm = self.add_pattern_to_lat(lat_pattern, lat_range_norm)
        
        lat_norm = np.array([lat_norm for _ in lon_range])
        lat_norm = lat_norm.transpose((1, 0, 2, 3))

        lon = np.array([
            np.arctan(
                kerX * sin_nu /
                (rho * np.cos(_lat) * cos_nu - kerY * np.sin(_lat) * sin_nu))
            for _lat in lat_range
        ])  # (W, Kh, Kw)

        lon_range_norm = self.min_max_norm(lon_range)
        lon_norm = np.array([lon + _lon for _lon in lon_range_norm])  # (W, H, Kh, Kw)
        lon_norm = lon_norm.transpose((1, 0, 2, 3))  # (H, W, Kh, Kw)

        lat = (lat_norm / 2 + 0.5) * self.coords_partial["x_total"]
        lon = ((lon_norm / 2 + 0.5) * self.coords_partial["y_total"])

        LatLon = np.stack(
            (lat, lon))  # (2, H, W, Kh, Kw) = ((lat, lon), H, W, Kh, Kw)
        LatLon = LatLon.transpose(
            (1, 3, 2, 4, 0))  # (H, Kh, W, Kw, 2) = (H, Kh, W, Kw, (lat, lon))
        # (35, 3, 35, 3, 2)
        H, Kh, W, Kw, d = LatLon.shape
        LatLon = LatLon.reshape((1, H * Kh, W * Kw, d))  # (1, H*Kh, W*Kw, 2)
        # (1, 105, 105, 2)
        return LatLon
        # return LatLon, lat_norm, lon_norm
    
    @classmethod
    def deal_with_coords_shrink(cls, out_w_end, out_w_start, out_w, shrink):
        assert shrink % 2 == 0, "the shrink size should be an even number for the double-side crop"
        if out_w == out_w_end + out_w_start:
            return out_w_end, out_w_start
        out_w_end = out_w_end - shrink // 2
        out_w_start = out_w_start - shrink // 2
        if out_w_start <= 0:
            out_w_end = out_w_end + out_w_start
            out_w_start = 0
        if out_w_end <= 0:
            out_w_start = out_w_start + out_w_end
            out_w_end = 0
        else:
            out_w_end, out_w_start = cls.deal_with_coords_shrink(out_w_end, out_w_start, out_w, shrink)
            return out_w_end, out_w_start

    def createKernel(self):
        """
    :return: (Ky, Kx) kernel pattern
    """

        Kh, Kw = self.kernel_size

        delta_lat = np.pi / self.coords_partial["x_total"]
        delta_lon = 2 * np.pi / self.coords_partial["y_total"]

        range_x = np.arange(-(Kw // 2), Kw // 2 + 1)
        if not Kw % 2:
            range_x = np.delete(range_x, Kw // 2)

        range_y = np.arange(-(Kh // 2), Kh // 2 + 1)
        if not Kh % 2:
            range_y = np.delete(range_y, Kh // 2)

        kerX = np.tan(range_x * delta_lon)
        kerY = np.tan(range_y * delta_lat) / np.cos(range_y * delta_lon)

        return np.meshgrid(kerX, kerY)  # (Kh, Kw)

    @classmethod
    def get_pattern(cls, angle, kernel_size):
        kh, kw = kernel_size
        pattern = np.empty_like(angle)
        for i, ker in enumerate(angle):
            # kernel_size
            center_value = ker[kh//2, kw//2]
            ker_1 = ker - center_value
            pattern[i] = ker_1
        return pattern
    
    @classmethod
    def add_pattern_to_lat(cls, pattern, lat_range):
        blocks = None
        for i, _lat in enumerate(lat_range):
            block = (_lat + pattern[i])[np.newaxis, :]
            if blocks is None:
                blocks = block
            else:
                blocks = np.concatenate([blocks, block], axis=0)
        return blocks
    
    @classmethod
    def min_max_norm(cls, x, start=-1):
        end = -start
        _len = end - start
        return (x - np.min(x)) / (np.max(x) - np.min(x)) * _len + start
    
    @classmethod
    def circular_norm(cls, lon_range_st, lon_range_ed):

        lon_sep = len(lon_range_st)
        lon_st_min = lon_range_st[0]
        lon_st_max = lon_range_st[-1]
        lon_ed_min = lon_range_ed[0]
        lon_ed_max = lon_range_ed[-1]
        lon_range = np.concatenate([lon_range_st, lon_range_ed], axis=0)
        lon_sep_coords = lon_sep / len(lon_range) * 2 - 1 
        if lon_st_max == lon_st_min:
            lon_range_st_norm = np.array([1])
        else:
            lon_range_st_norm = (lon_range_st - lon_st_min) / (lon_st_max - lon_st_min) * (lon_sep_coords + 1) - 1
        if lon_ed_min == lon_ed_max:
            lon_range_ed_norm = np.array([-1])
        else:
            lon_range_ed_norm = (lon_range_ed - lon_ed_min) / (lon_ed_max - lon_ed_min) * (1 - lon_sep_coords) + lon_sep_coords
        
        lon = np.concatenate((lon_range_st_norm, lon_range_ed_norm), axis=0)
        return lon

    @classmethod
    def convert_numpy(cls, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
            return x
        else:
            return x


class IncreIntervalGridGenerator:
    def __init__(self, height: int, width: int, kernel_size, stride=1, upsample=False):
        self.height = height
        self.width = width
        self.upsample = upsample
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)  # (Kh, Kw)
        else:
            self.kernel_size = kernel_size
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:  
            self.stride = stride  # (H, W)

    def createSamplingPattern(self):
        if self.upsample:
            kerX, kerY = self.createKernel()  # (Kh, Kw)
            # kerX: (3, 3)
            # create some values using in generating lat/lon sampling pattern
            rho = np.sqrt(kerX**2 + kerY**2)
            Kh, Kw = self.kernel_size
            # when the value of rho at center is zero, some lat values explode to `nan`.
            if Kh % 2 and Kw % 2:
                rho[Kh // 2][Kw // 2] = 1e-8

            nu = np.arctan(rho)
            cos_nu = np.cos(nu)
            sin_nu = np.sin(nu)

            stride_h, stride_w = self.stride
            
            # 2 * (I - padding - 1) + kernel + padding
            out_h = stride_h * (self.height - Kh * stride_h * 2 - 1) + (1 + stride_h * 2) * Kh
            out_w = stride_w * (self.width - Kw * stride_w * 2 - 1) + (1 + stride_w * 2) * Kw
            
            h_range = np.linspace(0, self.height, out_h)
            w_range = np.linspace(0, self.width, out_w)
            
            lat_range = ((h_range / self.height) - 0.5) * np.pi # lat_range: (1, height/stride)
            lon_range = ((w_range / self.width) - 0.5) * (2 * np.pi)

            # generate latitude sampling pattern
            lat = np.array([
                np.arcsin(cos_nu * np.sin(_lat) +
                        kerY * sin_nu * np.cos(_lat) / rho) for _lat in lat_range
            ])  # (H, Kh, Kw)
            lat = np.array([lat for _ in lon_range]) 
            lat = lat.transpose((1, 0, 2, 3))  # (H, W, Kh, Kw)

            lon = np.array([
                np.arctan(
                    kerX * sin_nu /
                    (rho * np.cos(_lat) * cos_nu - kerY * np.sin(_lat) * sin_nu))
                for _lat in lat_range
            ])  # (H, Kh, Kw)

            lon = np.array([lon + _lon for _lon in lon_range])  # (W, H, Kh, Kw)
            lon = lon.transpose((1, 0, 2, 3))  # (H, W, Kh, Kw)
            lat = (lat / np.pi + 0.5) * self.height
            lon = ((lon / (2 * np.pi) + 0.5) * self.width) % self.width

            LatLon = np.stack(
                (lat, lon))  # (2, H, W, Kh, Kw) = ((lat, lon), H, W, Kh, Kw)
            LatLon = LatLon.transpose(
                (1, 3, 2, 4, 0))  # (H, Kh, W, Kw, 2) = (H, Kh, W, Kw, (lat, lon))
            # (35, 3, 35, 3, 2)
            H, Kh, W, Kw, d = LatLon.shape
            LatLon = LatLon.reshape((1, H * Kh, W * Kw, d))  # (1, H*Kh, W*Kw, 2)
            # (1, 105, 105, 2)
            return LatLon
            
        else:
            kerX, kerY = self.createKernel()  # (Kh, Kw)
            # kerX: (3, 3)
            # create some values using in generating lat/lon sampling pattern
            rho = np.sqrt(kerX**2 + kerY**2)
            Kh, Kw = self.kernel_size
            # when the value of rho at center is zero, some lat values explode to `nan`.
            if Kh % 2 and Kw % 2:
                rho[Kh // 2][Kw // 2] = 1e-8

            nu = np.arctan(rho)
            cos_nu = np.cos(nu)
            sin_nu = np.sin(nu)
            
            stride_h, stride_w = self.stride
            if self.kernel_size[0] == 1:
                h_range = np.arange(0, self.height, stride_h)
                w_range = np.arange(0, self.width, stride_w)
            else:
                deleteX = self.kernel_size[0] // 2
                deleteY = self.kernel_size[1] // 2
                if self.kernel_size[0] % 2 == 0:
                    if stride_h == 1:
                        h_range = np.arange(0, self.height, stride_h)[deleteX-1: -deleteX]
                        h_range = np.linspace(0, self.height, len(h_range))
                    elif stride_h == 2:
                        h_range = np.arange(0, self.height, stride_h)[deleteX-1: -deleteX]
                        h_range = np.linspace(0, self.height, len(h_range))
                    else:
                        raise NotImplementedError
                else:
                    if stride_h == 1:
                        h_range = np.arange(0, self.height, stride_h)[deleteX: -deleteX]
                        h_range = np.linspace(0, self.height, len(h_range))
                    elif stride_h == 2:
                        if deleteX == 1:
                            h_range = np.arange(0, self.height, stride_h)
                            h_range = np.linspace(0, self.height, len(h_range))
                        else:
                            h_range = np.arange(0, self.height, stride_h)[deleteX-1: -deleteX+1]
                            h_range = np.linspace(0, self.height, len(h_range))
                    else:
                        raise NotImplementedError
                if self.kernel_size[1] % 2 == 0:
                    if stride_w == 1:
                        w_range = np.arange(0, self.width, stride_w)[deleteY-1: -deleteY]
                        w_range = np.linspace(0, self.width, len(w_range))
                    elif stride_w == 2:
                        w_range = np.arange(0, self.width, stride_w)[deleteY-1: -deleteY]
                        w_range = np.linspace(0, self.width, len(w_range))
                    else:
                        raise NotImplementedError
                else:
                    if stride_w == 1:
                        w_range = np.arange(0, self.width, stride_w)[deleteY: -deleteY]
                        w_range = np.linspace(0, self.width, len(w_range))
                    elif stride_w == 2:
                        if deleteY == 1:
                            w_range = np.arange(0, self.width, stride_w)
                            w_range = np.linspace(0, self.width, len(w_range))
                        else:
                            w_range = np.arange(0, self.width, stride_w)[deleteY-1: -deleteY+1]
                            w_range = np.linspace(0, self.width, len(w_range))
                    else:
                        raise NotImplementedError            

            lat_range = ((h_range / self.height) - 0.5) * np.pi # lat_range: (1, height/stride)
            lon_range = ((w_range / self.width) - 0.5) * (2 * np.pi)

            # generate latitude sampling pattern
            lat = np.array([
                np.arcsin(cos_nu * np.sin(_lat) +
                        kerY * sin_nu * np.cos(_lat) / rho) for _lat in lat_range
            ])  # (H, Kh, Kw)
            lat = np.array([lat for _ in lon_range])
            lat = lat.transpose((1, 0, 2, 3))  # (H, W, Kh, Kw)

            lon = np.array([
                np.arctan(
                    kerX * sin_nu /
                    (rho * np.cos(_lat) * cos_nu - kerY * np.sin(_lat) * sin_nu))
                for _lat in lat_range
            ])  # (H, Kh, Kw)

            lon = np.array([lon + _lon for _lon in lon_range])  # (W, H, Kh, Kw)
            lon = lon.transpose((1, 0, 2, 3))  # (H, W, Kh, Kw)
            # (radian) -> (index of pixel)
            lat = (lat / np.pi + 0.5) * self.height
            lon = ((lon / (2 * np.pi) + 0.5) * self.width) % self.width

            LatLon = np.stack(
                (lat, lon))  # (2, H, W, Kh, Kw) = ((lat, lon), H, W, Kh, Kw)
            LatLon = LatLon.transpose(
                (1, 3, 2, 4, 0))  # (H, Kh, W, Kw, 2) = (H, Kh, W, Kw, (lat, lon))
            # (35, 3, 35, 3, 2)
            H, Kh, W, Kw, d = LatLon.shape
            LatLon = LatLon.reshape((1, H * Kh, W * Kw, d))  # (1, H*Kh, W*Kw, 2)
            # (1, 105, 105, 2)
            return LatLon

    def createKernel(self):
        """
    :return: (Ky, Kx) kernel pattern
    """

        Kh, Kw = self.kernel_size

        delta_lat = np.pi / self.height
        delta_lon = 2 * np.pi / self.width

        range_x = np.arange(-(Kw // 2), Kw // 2 + 1)
        if not Kw % 2:
            range_x = np.delete(range_x, Kw // 2)

        range_y = np.arange(-(Kh // 2), Kh // 2 + 1)
        if not Kh % 2:
            range_y = np.delete(range_y, Kh // 2)

        kerX = np.tan(range_x * delta_lon) # (3, )
        kerY = np.tan(range_y * delta_lat) / np.cos(range_y * delta_lon) # (3, )

        return np.meshgrid(kerX, kerY)  # (Kh, Kw)


class GridSampler(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, z, grid):
        # return F.grid_sample(z, grid, align_corners=True, mode='nearest')
        return grid_sample(z, grid)


class GridSamplerNew(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, z, grid):
        return F.grid_sample(z, grid, align_corners=True, mode='bilinear', padding_mode="border")


class GridSamplerNewTexture(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, z, grid):
        return grid_sample_github(z, grid)

        
class GridSamplerNewTextureNoGrad(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, z, grid):
        return GridSamplerFuncNoGrad.apply(z, grid)


class GridSamplerFuncNoGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, grid):
        result = F.grid_sample(z, grid, align_corners=True, mode='bilinear', padding_mode='border')
        return result

    @staticmethod
    def backward(ctx, grad_output):
        B, C, H, W = grad_output.shape
        reshaped_tensor = grad_output.contiguous().reshape(B, C, H//3, 3, W//3, 3)
        averaged_tensor = reshaped_tensor.mean(dim=[3, 5])
        grad_input = averaged_tensor * 0.1
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(grad_input)
        return grad_input, None
