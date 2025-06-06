import torch
import random
import numpy as np


class LatentSampler():
    constant_global_latent = None
    constant_local_latent = None
    def __init__(self, generator, config):
        self.config = config
        self.generator = generator

    @torch.no_grad()
    def sample_global_latent(self, batch_size, device, requires_grad=False, mixing=True, seed=None):
        global_latent_dim = self.config.train_params.global_latent_dim
        is_mixing = random.random() < self.config.train_params.mixing if mixing else False

        if seed is None:
            latent_1 = torch.randn(batch_size, global_latent_dim, device=device)
            latent_2 = torch.randn(batch_size, global_latent_dim, device=device)
        else:
            latent_1 = torch.from_numpy(np.random.RandomState(seed).randn((batch_size, global_latent_dim))).to(device)
            latent_2 = torch.from_numpy(np.random.RandomState(seed).randn((batch_size, global_latent_dim))).to(device)
        latent = torch.stack([
            latent_1,
            latent_2 if is_mixing else latent_1,
        ], 1) # shape: (B, 2, D) # batch-first for dataparallel

        latent.requires_grad = requires_grad
        return latent
    
    @torch.no_grad()
    def sample_constant_global_latent(self, batch_size, device, requires_grad=False, mixing=True):
        if self.constant_global_latent is None:
            global_latent_dim = self.config.train_params.global_latent_dim
            is_mixing = random.random() < self.config.train_params.mixing if mixing else False

            latent_1 = torch.randn(batch_size, global_latent_dim, device=device)
            latent_2 = torch.randn(batch_size, global_latent_dim, device=device)
            latent = torch.stack([
                latent_1,
                latent_2 if is_mixing else latent_1,
            ], 1) # shape: (B, 2, D) # batch-first for dataparallel

            latent.requires_grad = requires_grad
            self.constant_global_latent = latent
            
        return self.constant_global_latent

    def sample_local_latent(self, batch_size, device, requires_grad=False,
                            spatial_size_enlarge=1, specific_shape=None, exclude_padding=False):

        local_latent_dim = self.config.train_params.local_latent_dim   

        if specific_shape is not None:
            if isinstance(specific_shape, int):
                spatial_shape = (specific_shape, specific_shape)
            else:
                spatial_shape = specific_shape
        elif spatial_size_enlarge != 1:
            if hasattr(self.config.train_params, "styleGAN2_baseline") and self.config.train_params.styleGAN2_baseline:
                size = self.config.train_params.ts_input_size * spatial_size_enlarge
                spatial_shape = (size, size)
            else:
                base = self.config.train_params.ts_input_size // 2
                size = (int(round(base * spatial_size_enlarge)) * 2) + 1
                spatial_shape = (size, size)
        else:
            size = self.config.train_params.ts_input_size
            spatial_shape = (size, size)
        
        if self.config.train_params.use_ss and self.config.train_params.ss_unfold_radius > 0:
            if self.config.train_params.ss_n_layers > 0:
                ss_unfold_size = self.config.train_params.ss_n_layers * self.config.train_params.ss_unfold_radius
            else:
                ss_unfold_size = 0
            if exclude_padding:
                spatial_shape_ext = spatial_shape
            else:
                spatial_shape_ext = [
                    spatial_shape[0] + 2 * ss_unfold_size,
                    spatial_shape[1] + 2 * ss_unfold_size]
            z_local = torch.randn(batch_size, local_latent_dim, spatial_shape_ext[0], spatial_shape_ext[1], device=device)
        else:
            z_local = torch.randn(batch_size, local_latent_dim, spatial_shape[0], spatial_shape[1], device=device)

        z_local.requires_grad = requires_grad
        return z_local

    def sample_constant_local_latent(self, batch_size, device, requires_grad=False,
                            spatial_size_enlarge=1, specific_shape=None, exclude_padding=False):
        if self.constant_local_latent is None:
            local_latent_dim = self.config.train_params.local_latent_dim   

            if specific_shape is not None:
                if isinstance(specific_shape, int):
                    spatial_shape = (specific_shape, specific_shape)
                else:
                    spatial_shape = specific_shape
            elif spatial_size_enlarge != 1:
                if hasattr(self.config.train_params, "styleGAN2_baseline") and self.config.train_params.styleGAN2_baseline:
                    size = self.config.train_params.ts_input_size * spatial_size_enlarge
                    spatial_shape = (size, size)
                else:
                    base = self.config.train_params.ts_input_size // 2
                    size = (int(round(base * spatial_size_enlarge)) * 2) + 1
                    spatial_shape = (size, size)
            else:
                size = self.config.train_params.ts_input_size
                spatial_shape = (size, size)
            
            if self.config.train_params.use_ss and self.config.train_params.ss_unfold_radius > 0:
                if self.config.train_params.ss_n_layers > 0:
                    ss_unfold_size = self.config.train_params.ss_n_layers * self.config.train_params.ss_unfold_radius
                else:
                    ss_unfold_size = 0
                if exclude_padding:
                    spatial_shape_ext = spatial_shape
                else:
                    spatial_shape_ext = [
                        spatial_shape[0] + 2 * ss_unfold_size,
                        spatial_shape[1] + 2 * ss_unfold_size]
                z_local = torch.randn(batch_size, local_latent_dim, spatial_shape_ext[0], spatial_shape_ext[1], device=device)
            else:
                z_local = torch.randn(batch_size, local_latent_dim, spatial_shape[0], spatial_shape[1], device=device)

            z_local.requires_grad = requires_grad
            self.constant_local_latent = z_local
        return self.constant_local_latent

    def sample_slicing_local_latent(self, batch_size, device, requires_grad=False,
                        spatial_size_enlarge=1, specific_shape=None, exclude_padding=False, padding_size=None):
        
        if padding_size is None:
            raise NotImplementedError(f"padding_size should be zero instead of {padding_size}")
        else:
            padding_shape = padding_size
        
        local_latent_dim = self.config.train_params.local_latent_dim   

        if specific_shape is not None:
            if isinstance(specific_shape, int):
                spatial_shape = (specific_shape, specific_shape)
            else:
                spatial_shape = specific_shape
        elif spatial_size_enlarge != 1:
            if hasattr(self.config.train_params, "styleGAN2_baseline") and self.config.train_params.styleGAN2_baseline:
                size = self.config.train_params.ts_input_size * spatial_size_enlarge
                spatial_shape = (size, size)
            else:
                base = self.config.train_params.ts_input_size // 2
                size = (int(round(base * spatial_size_enlarge)) * 2) + 1
                spatial_shape = (size, size)
        else:
            size = self.config.train_params.ts_input_size
            spatial_shape = (size, size)
        
        spatial_shape = (
            spatial_shape[0] + padding_shape,
            spatial_shape[1] + padding_shape
            )
        
        if self.config.train_params.use_ss and self.config.train_params.ss_unfold_radius > 0:
            if self.config.train_params.ss_n_layers > 0:
                ss_unfold_size = self.config.train_params.ss_n_layers * self.config.train_params.ss_unfold_radius
            else:
                ss_unfold_size = 0
            if exclude_padding:
                spatial_shape_ext = spatial_shape
            else:
                spatial_shape_ext = [
                    spatial_shape[0] + 2 * ss_unfold_size,
                    spatial_shape[1] + 2 * ss_unfold_size]
            z_local = torch.randn(batch_size, local_latent_dim, spatial_shape_ext[0], spatial_shape_ext[1], device=device)
        else:
            z_local = torch.randn(batch_size, local_latent_dim, spatial_shape[0], spatial_shape[1], device=device)

        z_local.requires_grad = requires_grad
        return z_local

    def sample_normal_spconv_local_latent(self, batch_size, device, requires_grad=False,
                        spatial_size_enlarge=1, specific_shape=None, exclude_padding=False):
        local_latent_dim = self.config.train_params.local_latent_dim   
        if specific_shape is not None:
            if isinstance(specific_shape, int):
                spatial_shape = (specific_shape, specific_shape)
            else:
                spatial_shape = specific_shape
        elif spatial_size_enlarge != 1:
            if hasattr(self.config.train_params, "styleGAN2_baseline") and self.config.train_params.styleGAN2_baseline:
                size = self.config.train_params.ts_input_size * spatial_size_enlarge
                spatial_shape = (size, size)
            else:
                base = self.config.train_params.ts_input_size // 2
                size = (int(round(base * spatial_size_enlarge)) * 2) + 1
                spatial_shape = (size, size)
        else:
            size = self.config.train_params.ts_input_size
            spatial_shape = (size, size)
        
        z_local = torch.randn(batch_size, local_latent_dim, spatial_shape[0], spatial_shape[1], device=device)
        z_local.requires_grad = requires_grad
        return z_local

    def sample_circular_local_latent_patch101(self, batch_size, device, meta_width, height_in, width_given=None, requires_grad=False, height_padding=True, padding_size=None, seed=None, step_size=None):
        local_latent_dim = self.config.train_params.local_latent_dim  
        if width_given is not None:
            width = meta_width // width_given * 6
        else:
            if step_size is None: 
                if meta_width == 1056:
                    width = meta_width // 96 * 6
                elif meta_width == 768 or  meta_width == 864 or  meta_width == 960:
                    width = meta_width // 96 * 6
                elif meta_width == 1152 or meta_width == 2112 or meta_width == 2880:
                    width = meta_width // 192 * 6
                elif meta_width <= 768 or meta_width % 192 == 0:
                    width = meta_width // 96 * 6
                else:
                    if width_given:
                        width = width_given
                    else:
                        raise NotImplementedError(f"meta width {meta_width} is not supportted")
            else:
                assert meta_width == 1152, f"other width {meta_width} is not supported"
                width = meta_width // 192 * step_size
                
        if padding_size:
            width += padding_size
            
        if self.config.train_params.ss_n_layers > 0:
            ss_unfold_size = self.config.train_params.ss_n_layers * self.config.train_params.ss_unfold_radius
        else:
            ss_unfold_size = 0

        if height_padding:
            height = height_in + 2 * ss_unfold_size
        else:
            height = height_in
        if seed is not None:
            z_local = torch.from_numpy(np.random.RandomState(seed).randn((batch_size, local_latent_dim, height, width))).to(device)
        else:
            z_local = torch.randn(batch_size, local_latent_dim, height, width, device=device)
        z_local.requires_grad = requires_grad
        return z_local
