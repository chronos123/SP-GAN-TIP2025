import os
import math
import datetime
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from itertools import product as iter_product

import torch

from test_managers.base_test_manager import BaseTestManager, BaseTestManagerNewInterval
from test_managers.base_slicing_test_manager import SlicingBaseTestManager
from test_managers.testing_vars_wrapper import TestingVars
from test_managers.global_config import test_meta_extra_pad
import logging


DEBUG_MODE = False
logging.basicConfig(
    filename="interval_test.log",
    filemode="w",
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

__logger__ = logging.getLogger(__name__)

# 用于baseline, 全部是普通卷积的生成器测试
class InfiniteGenerationManagerCloseLoop(BaseTestManager):

    def task_specific_init(self, output_size=None):

        if output_size is None:
            self.target_height = self.config.task.height
            self.target_width = self.config.task.width
        else:
            self.target_height, self.target_width = output_size
           
        self._init_starting_points()         
        self.noise_heights = self.outfeat_step_sizes * (self.num_steps_h-1) + self.outfeat_sizes_list
        self.noise_widths  = self.outfeat_step_sizes * (self.num_steps_w_min)

        if hasattr(self.config.task, "gen_from_inv_stats") and self.config.task.gen_from_inv_stats:
            self.inv_root = self.compose_inv_root()
            self.inv_rec_files = sorted(glob(os.path.join(self.inv_root, "*")))
            self.gen_from_inv_stats = True
        else:
            self.gen_from_inv_stats = False

    def run_next(self, save=True, write_gpu_time=False, inv_records=None, inv_placements=None, calc_flops=False, disable_pbar=False, **kwargs):
        if len(kwargs) > 0:
            for k,v in kwargs.items():
                if v is not None:
                    print(" [Warning] task manager receives untracked arg {} with value {}".format(k ,v))
        testing_vars = self.create_vars(inv_records=inv_records, inv_placements=inv_placements)
        self.generate(testing_vars, write_gpu_time=write_gpu_time, calc_flops=calc_flops, disable_pbar=disable_pbar)
        if save:
            self.save_results(testing_vars.meta_img)
        return testing_vars.meta_img

    def create_vars(self, inv_records=None, inv_placements=None, seed=None):
        # 对local_latent进行采样
        mixing = False
        assert mixing == False, "Otherwise, an injection index must be specified and fed into g_ema."

        # Allocate memory for the final output, starts iterating and filling in the generated results.
        # Can be reused
        meta_img = torch.empty(
            self.config.train_params.batch_size,
            3,
            int(self.meta_height),
            int(self.meta_width)).float()

        # [Note]
        # 1.  One may implement a sophisticated version that does not required to 
        #     generate all the latents at once, as most of the info are not reusing
        #     during the inference. However, the author is just lazy and abusing his 
        #     CPU memory OuO
        global_latent = self.latent_sampler.sample_global_latent(
            self.config.train_params.batch_size, mixing=mixing, device=self.device, seed=seed)
        # full_local_latent_shape = (
        #     # Does not account GNN padding here, it is handled within the latent_sampler
        #     int(self.g_ema_module.calc_in_spatial_size(self.meta_height, include_ss=False)),
        #     int(self.g_ema_module.calc_in_spatial_size(self.meta_width, include_ss=False)),
        # )
        # local_latent = self.latent_sampler.sample_local_latent(
        #     self.config.train_params.batch_size, 
        #     device="cpu", # Store in CPU anyway, it can be INFINITLY LARGE!
        #     specific_shape=full_local_latent_shape)

        height = self.g_ema_module.calc_in_spatial_size(self.meta_height, include_ss=False)

        local_latent = self.latent_sampler.sample_circular_local_latent_patch101(
            self.config.train_params.batch_size,
            device="cpu",
            meta_width=self.meta_width,
            height_in=height,
            seed=seed,
        )

        meta_coords = self.coord_handler.sample_coord_grid(
            local_latent, 
            is_training=False) # 通过local_latent得到坐标

        # Create randomized noises, 得到随机噪声
        randomized_noises = [
            torch.randn(self.config.train_params.batch_size, 1, int(h), int(w))
                for (h,w) in zip(self.noise_heights, self.noise_widths)]

        testing_vars = TestingVars(
            meta_img=meta_img, 
            global_latent=global_latent, 
            local_latent=local_latent, 
            meta_coords=meta_coords, 
            noises=randomized_noises, 
            device=self.device)
        # 通过testing vars对测试的参数进行封装

        if self.gen_from_inv_stats:
            assert inv_records is None, \
                "`gen_from_inv_stats` already specified, should not receive `inv_records` from command!"
            assert self.config.train_params.batch_size == 1, \
                "Inverted parameters loading for batch is not yet implemeted! " + \
                "Please use parallel-batching instead, which provides a similar inference speed."
            inv_records = [self.inv_rec_files[self.cur_global_id]]
            inv_placements = [self.config.task.gen_from_inv_placement]

        if inv_records is not None:
            testing_vars.replace_by_records(
                self.g_ema_module, inv_records, inv_placements, assert_no_style=True)
        
        return testing_vars
    
    def circular_sample_width(self, tensor:torch.Tensor, y_width, x_st, x_ed, y_st, y_ed):
        """_summary_

        Args:
            tensor (torch.Tensor): value to be sampled
            y_width (int): total_width of the tensor to be sampled
            y_st (int): start point of the slice (non-circular)
            y_ed (int): end point of the slice (non-circular: may be beyond the x_width)
        """
        if y_ed <= y_width:
            # 不超出循环的情况
            return tensor[:, :, x_st:x_ed, y_st:y_ed]
        elif y_ed <= y_width * 2:
            # 超出循环一倍以内的情况
            if y_st < y_width:
                y_ed = y_ed % y_width
                return torch.cat((tensor[:, :, x_st:x_ed, y_st:], tensor[:, :, x_st:x_ed, :y_ed]), dim=3)
            else:
                y_st = y_st % y_width
                y_ed = y_ed % y_width
                return tensor[:, :, x_st:x_ed, y_st:y_ed]
        else:
            # 超出循环两倍及以上进行递归求解
            assert y_st > y_width
            y_st = y_st - y_width
            y_ed = y_ed - y_width
            return self.circular_sample_width(tensor, y_width, x_st, x_ed, y_st, y_ed)

    def generate(self, testing_vars, tkinter_pbar=None, update_by_ss_map=None, update_by_ts_map=None, 
                 write_gpu_time=False, calc_flops=False, disable_pbar=False):

        # I don't mind bruteforce casting combination here, cuz you should worry about the meta_img size first
        idx_tuples = list(iter_product(range(self.start_pts_mesh_z.shape[0]), range(self.start_pts_mesh_z.shape[1])))

        if disable_pbar:
            pbar = idx_tuples
        elif tkinter_pbar is not None:
            pbar = tkinter_pbar(idx_tuples)
        else:
            pbar = tqdm(idx_tuples)

        accum_exec_time = 0
        accum_flops_all, accum_flops_ss, accum_flops_ts = 0, 0, 0
        for iiter, (idx_x,idx_y) in enumerate(pbar):
            zx_st, zy_st = self.start_pts_mesh_z[idx_x, idx_y]
            zx_ed = zx_st + self.config.train_params.ts_input_size 
            zy_ed = zy_st + self.config.train_params.ts_input_size

            # Handle the randomized noise input of the texture_synthesizer...
            outfeat_x_st = [start_pts_mesh[idx_x,idx_y,0] for start_pts_mesh in self.start_pts_mesh_outfeats]
            outfeat_y_st = [start_pts_mesh[idx_x,idx_y,1] for start_pts_mesh in self.start_pts_mesh_outfeats]
            outfeat_x_ed = [
                x_st + out_size for (x_st, out_size) in zip(outfeat_x_st, self.outfeat_sizes_list)]
            outfeat_y_ed = [
                y_st + out_size for (y_st, out_size) in zip(outfeat_y_st, self.outfeat_sizes_list)]
            noises = []
            for i, (fx_st, fy_st, fx_ed, fy_ed) in enumerate(zip(outfeat_x_st, outfeat_y_st, outfeat_x_ed, outfeat_y_ed)):
                # 实现循环采样
                noises.append(self.circular_sample_width(testing_vars.noises[i], self.noise_widths[i], fx_st, fx_ed, fy_st, fy_ed).to(self.device))
            # 0:19, 0:17, 0:31, 0:29, 0:55, 0:53, 0:103, 0:101; x,y相同, 由start_pts_mesh控制
            # x相同, y为 12:31, 12:29, 24:55, 24:53, 48:103, 48:101, 96:199, 96:197  
            # noise 到 meta_height x meta_width, 和local_latent相对应, 这里偏置为12
            # local_latent的偏置为6 
            # Deal with SS unfolding here
            zx_st -= self.ss_unfold_size
            zy_st -= self.ss_unfold_size
            zx_ed += self.ss_unfold_size
            zy_ed += self.ss_unfold_size
            
            # [Interactive] Decide whether the region will be updated, otherwise no need to generate
            if update_by_ss_map is not None:
                ss_cursors = zx_st, zx_ed, zy_st, zy_ed
                if not self.is_overlaping_update_map(update_by_ss_map, *ss_cursors):
                    continue
            if update_by_ts_map is not None:
                # For TS regional selection, we only select noises
                ts_cursors = outfeat_x_st[0], outfeat_x_ed[0], outfeat_y_st[0], outfeat_y_ed[0]
                if not self.is_overlaping_update_map(update_by_ts_map, *ts_cursors):
                    continue
            
            # 实现循环采样
            cur_local_latent = self.circular_sample_width(testing_vars.local_latent, testing_vars.local_latent.shape[3], zx_st, zx_ed, zy_st, zy_ed).to(self.device)
            # 0:35, 0:35; 下一循环为 0:35, 6:41

            # 实现循环采样
            cur_coords = self.circular_sample_width(testing_vars.meta_coords, testing_vars.meta_coords.shape[3], zx_st, zx_ed, zy_st, zy_ed).to(self.device)
            g_ema_kwargs = {
                "global_latent": testing_vars.global_latent,
                "local_latent": cur_local_latent,
                "override_coords": cur_coords,
                "noises": noises,
                "disable_dual_latents": True,
                "calc_flops": calc_flops,
            }
            # 得到上面这些参数就可以生成小块
            if hasattr(testing_vars, "wplus_styles") and testing_vars.wplus_styles is not None:
                g_ema_kwargs["wplus_styles"] = testing_vars.wplus_styles
            img_x_st, img_y_st = outfeat_x_st[-1], outfeat_y_st[-1]
            img_x_ed, img_y_ed = outfeat_x_ed[-1], outfeat_y_ed[-1]
            index_tuple = (img_x_st, img_x_ed, img_y_st, img_y_ed) # 0-101, 0-101
            # 下一句生成小块
            exec_time, flops = self.maybe_parallel_inference(
                testing_vars, g_ema_kwargs=g_ema_kwargs, index_tuple=index_tuple, return_exec_time=write_gpu_time, calc_flops=calc_flops)
            accum_exec_time += exec_time
            if calc_flops:
                accum_flops_all += flops["all"]
                accum_flops_ss += flops["ss"]
                accum_flops_ts += flops["ts"]

        exec_time, flops = self.maybe_parallel_inference(
            testing_vars, flush=True, return_exec_time=write_gpu_time, calc_flops=calc_flops)
        if calc_flops:
            accum_flops_all += flops["all"]
            accum_flops_ss += flops["ss"]
            accum_flops_ts += flops["ts"]
        
        if write_gpu_time:
            accum_exec_time += exec_time
            print(" [*] GPU time {}".format(accum_exec_time))
            self.accum_exec_times.append(accum_exec_time)
            fmt_date = datetime.date.today().strftime("%d-%m-%Y")
            benchmark_file = os.path.join(self.save_root, "speed_benchmark_{}.txt".format(fmt_date))
            with open(benchmark_file, "a") as f:
                f.write("{:.6f}".format(accum_exec_time))

        if calc_flops:
            print(" [*] Total FLOPs: {} (SS {}, TS {})".format(
                self.pretty_print_flops(accum_flops_all), 
                self.pretty_print_flops(accum_flops_ss), 
                self.pretty_print_flops(accum_flops_ts)))

    def save_results(self, meta_img, dump_vars=None):
        print(" [*] Saving results...")
        self.save_meta_imgs(meta_img)
        if dump_vars is not None:
            self.save_testing_vars(dump_vars)
        self.cur_global_id += self.config.train_params.batch_size

    def save_testing_vars(self, testing_vars):
        assert self.config.train_params.batch_size == 1, \
            "This is only designed to be used with the interactive tool."
        save_path = os.path.join(self.save_root, str(self.cur_global_id).zfill(6)+".pkl")
        pkl.dump(testing_vars, open(save_path, "wb"))

    def _wrap_feature(self, feat, wrap_size, dim):
        assert wrap_size < (feat.shape[dim] - 2*wrap_size), \
            "Does not expect the wrapping area is larger than a full period."
        if dim == 2:
            valid_st = feat[:, :, wrap_size:2*wrap_size]
            valid_ed = feat[:, :, -2*wrap_size:-wrap_size]
            feat[:, :, :wrap_size] = valid_ed
            feat[:, :, -wrap_size:] = valid_st
        elif dim == 3:
            valid_st = feat[:, :, :, wrap_size:2*wrap_size]
            valid_ed = feat[:, :, :, -2*wrap_size:-wrap_size]
            feat[:, :, :, :wrap_size] = valid_ed
            feat[:, :, :, -wrap_size:] = valid_st
        else:
            raise NotImplementedError(
                "I don't expect this function will be used other than spatial dims, but got {}.".format(dim))
        
    def save_meta_imgs(self, meta_img):
        self.full_image = meta_img
        # Center crop
        pad_h = (self.meta_height - self.target_height) // 2
        pad_w = (self.meta_width - self.target_width) // 2
        meta_img = meta_img[:, :, pad_h:pad_h+self.target_height, pad_w:pad_w+self.target_width]

        # Save the full image and the low-resolution image (for visualization)
        meta_img = meta_img.clamp(-1, 1).permute(0, 2, 3, 1)
        meta_img = (meta_img + 1) / 2
        meta_img_np = meta_img.numpy()
        
        for i in range(self.config.train_params.batch_size):
            global_id = self.cur_global_id + i
            save_path = os.path.join(self.save_root, str(global_id).zfill(6)+".png")
            plt.imsave(save_path, meta_img_np[i])

    def save_full_imgs(self):
        # Save the full image and the low-resolution image (for visualization)
        meta_img = self.full_image
        meta_img = meta_img.clamp(-1, 1).permute(0, 2, 3, 1)
        meta_img = (meta_img + 1) / 2
        meta_img_np = meta_img.numpy()
        
        for i in range(self.config.train_params.batch_size):
            global_id = self.cur_global_id + i - self.config.train_params.batch_size
            # save after adding the global_id
            save_path = os.path.join(self.save_root, str(global_id).zfill(6)+"full.png")
            plt.imsave(save_path, meta_img_np[i])

    def _create_start_pts_mesh(self, step_size, num_steps_h, num_steps_w):
        # 定义重叠的尺寸
        start_pts_x = np.arange(num_steps_h) * step_size
        start_pts_y = np.arange(num_steps_w) * step_size
        start_pts_mesh = np.stack([
            np.repeat(start_pts_x.reshape(num_steps_h, 1), num_steps_w, axis=1),
            np.repeat(start_pts_y.reshape(1, num_steps_w), num_steps_h, axis=0),
        ], 2).astype(np.uint32) # shape: (H, W, 2)
        return start_pts_mesh

    def _init_starting_points(self):
        # 定义重叠的尺寸
        # Get all the starting points of the patches to generate, 计算生成的小块数量
        self.num_steps_h = \
            math.ceil((self.target_height - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad
        # self.num_steps_w = \
        #     math.ceil((self.target_width  - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad
        
        self.num_steps_w_min = math.ceil(self.target_width / self.pixelspace_step_size)
        self.num_steps_w = self.num_steps_w_min + 5
        # 多出的步数为重叠回来的步数
        
        self.start_pts_mesh_z = self._create_start_pts_mesh(
            step_size=self.latentspace_step_size, 
            num_steps_h=self.num_steps_h,
            num_steps_w=self.num_steps_w)
        self.start_pts_mesh_z += self.ss_unfold_size

        # Create this for:
        # (1) Final image pixels assignment
        # (2) Randomized noise handling within the texture synthesizer
        self.start_pts_mesh_outfeats = [
            self._create_start_pts_mesh(
                step_size=step_size,
                num_steps_h=self.num_steps_h,
                num_steps_w=self.num_steps_w,
            ) for step_size in self.outfeat_step_sizes]
        # start_pts_mesh_x = \
        #     (start_pts_mesh_z - ss_unfold_size) // latentspace_step_size * pixelspace_step_size # shape: (H, W, 2)
        # start_pts_mesh_x = start_pts_mesh_x.astype(np.uint32)

        # To avoid edge-condition on the image edge, we generate an image slightly larger than
        # requested, then center-crop to the requested resolution.
        self.meta_height = self.pixelspace_step_size * (self.num_steps_h-1) + self.outfeat_sizes_list[-1]
        self.meta_width  = self.num_steps_w_min * self.pixelspace_step_size
        # height=485, width=773 (256 x 512)
        # height=485, width=965 (256 x 768)

    def compose_inv_root(self):
        return os.path.join("./logs/", self.config.var.exp_name, "test", self.config.task.prev_inv_config, "stats")


# patch coords 用于球卷积的生成器测试
class InfiniteGenerationManagerPatchCoordsCloseLoop(BaseTestManager):
    attention_flag = False
    def save_full_imgs(self):
        # Save the full image and the low-resolution image (for visualization)
        meta_img = self.full_image
        meta_img = meta_img.clamp(-1, 1).permute(0, 2, 3, 1)
        meta_img = (meta_img + 1) / 2
        meta_img_np = meta_img.numpy()
        
        for i in range(self.config.train_params.batch_size):
            global_id = self.cur_global_id + i - self.config.train_params.batch_size
            # save after adding the global_id
            save_path = os.path.join(self.save_root, str(global_id).zfill(6)+"full.png")
            plt.imsave(save_path, meta_img_np[i])
            
    def task_specific_init(self, output_size=None, **kwargs):

        if output_size is None:
            self.target_height = self.config.task.height
            self.target_width = self.config.task.width
        else:
            self.target_height, self.target_width = output_size
           
        self._init_close_loop_starting_points()         
        self.noise_heights = self.outfeat_step_sizes * (self.num_steps_h-1) + self.outfeat_sizes_list
        self.noise_widths  = self.outfeat_step_sizes * self.num_steps_w_min

        if hasattr(self.config.task, "gen_from_inv_stats") and self.config.task.gen_from_inv_stats:
            self.inv_root = self.compose_inv_root()
            self.inv_rec_files = sorted(glob(os.path.join(self.inv_root, "*")))
            self.gen_from_inv_stats = True
        else:
            self.gen_from_inv_stats = False

        self.save = False
        # print("save all random noises")
        # self.save_dict = {}
        # self.latent_save_path = os.path.join(self.save_root, "latents")
        # os.makedirs(self.latent_save_path, exist_ok=True)

    def run_next(self, save=True, write_gpu_time=False, inv_records=None, inv_placements=None, calc_flops=False, disable_pbar=False, **kwargs):
        if len(kwargs) > 0:
            for k,v in kwargs.items():
                if v is not None and k not in ["seeds"]:
                    print(" [Warning] task manager receives untracked arg {} with value {}".format(k ,v))
        testing_vars = self.create_vars(inv_records=inv_records, inv_placements=inv_placements, seed=kwargs.get("seeds", None))
        self.generate(testing_vars, write_gpu_time=write_gpu_time, calc_flops=calc_flops, disable_pbar=disable_pbar)
        if save:
            self.save_results(testing_vars.meta_img)
        return testing_vars.meta_img

    def create_vars(self, inv_records=None, inv_placements=None, seed=None):
        # 对local_latent进行采样
        mixing = False
        assert mixing == False, "Otherwise, an injection index must be specified and fed into g_ema."
        
        # Allocate memory for the final output, starts iterating and filling in the generated results.
        # Can be reused
        meta_img = torch.empty(
            self.config.train_params.batch_size,
            3,
            int(self.meta_height),
            int(self.meta_width)).float()

        # [Note]
        # 1.  One may implement a sophisticated version that does not required to 
        #     generate all the latents at once, as most of the info are not reusing
        #     during the inference. However, the author is just lazy and abusing his 
        #     CPU memory OuO
        global_latent = self.latent_sampler.sample_global_latent(
            self.config.train_params.batch_size, mixing=mixing, device=self.device, seed=seed)
        # full_local_latent_shape = (
        #     # Does not account GNN padding here, it is handled within the latent_sampler
        #     int(self.g_ema_module.calc_in_spatial_size(self.meta_height, include_ss=False)),
        #     int(self.g_ema_module.calc_in_spatial_size(self.meta_width, include_ss=False)),
        # )
        # local_latent = self.latent_sampler.sample_local_latent(
        #     self.config.train_params.batch_size, 
        #     device="cpu", # Store in CPU anyway, it can be INFINITLY LARGE!
        #     specific_shape=full_local_latent_shape)

        height = self.g_ema_module.calc_in_spatial_size(self.meta_height, include_ss=False)
        width_given = self.infeat_step_sizes[-1]
        local_latent = self.latent_sampler.sample_circular_local_latent_patch101(
            self.config.train_params.batch_size,
            device="cpu",
            meta_width=self.meta_width,
            height_in=height,
            width_given=width_given,
            seed=seed,
        )

        self.full_shape = local_latent.shape[2:]
        if hasattr(self.config.task, "force_inside") and self.config.task.force_inside:
            meta_coords = self.coord_handler.sample_coord_grid(
                local_latent, 
                is_training=False,
                force_inside=self.config.task.force_inside,
                ) # 通过local_latent得到坐标
        else:
            meta_coords = self.coord_handler.sample_coord_grid(
                local_latent, 
                is_training=False,
                ) # 通过local_latent得到坐标

        # Create randomized noises, 得到随机噪声
        # 噪声也由seed控制，保证一样
        # if seed is not None:
        #     randomized_noises = [
        #         torch.from_numpy(np.random.RandomState(seed).randn(self.config.train_params.batch_size, 1, int(h), int(w)))
        #             for (h,w) in zip(self.noise_heights, self.noise_widths)]
        # else:
        randomized_noises = [
            torch.randn(self.config.train_params.batch_size, 1, int(h), int(w))
                for (h,w) in zip(self.noise_heights, self.noise_widths)]

        if self.save:
            self.save_dict["local_latent"] = local_latent
            self.save_dict["global_latent"] = global_latent
            self.save_dict["noises"] = randomized_noises

        testing_vars = TestingVars(
            meta_img=meta_img, 
            global_latent=global_latent, 
            local_latent=local_latent, 
            meta_coords=meta_coords, 
            noises=randomized_noises, 
            device=self.device)
        # 通过testing vars对测试的参数进行封装

        if self.gen_from_inv_stats:
            assert inv_records is None, \
                "`gen_from_inv_stats` already specified, should not receive `inv_records` from command!"
            assert self.config.train_params.batch_size == 1, \
                "Inverted parameters loading for batch is not yet implemeted! " + \
                "Please use parallel-batching instead, which provides a similar inference speed."
            inv_records = [self.inv_rec_files[self.cur_global_id]]
            inv_placements = [self.config.task.gen_from_inv_placement]

        if inv_records is not None:
            testing_vars.replace_by_records(
                self.g_ema_module, inv_records, inv_placements, assert_no_style=True)
        
        return testing_vars

    def generate(self, testing_vars, tkinter_pbar=None, update_by_ss_map=None, update_by_ts_map=None, 
                 write_gpu_time=False, calc_flops=False, disable_pbar=False):

        # I don't mind bruteforce casting combination here, cuz you should worry about the meta_img size first
        idx_tuples = list(iter_product(range(self.start_pts_mesh_z.shape[0]), range(self.start_pts_mesh_z.shape[1])))

        if disable_pbar:
            pbar = idx_tuples
        elif tkinter_pbar is not None:
            pbar = tkinter_pbar(idx_tuples)
        else:
            pbar = tqdm(idx_tuples)

        accum_exec_time = 0
        accum_flops_all, accum_flops_ss, accum_flops_ts = 0, 0, 0
        for iiter, (idx_x,idx_y) in enumerate(pbar):
            zx_st, zy_st = self.start_pts_mesh_z[idx_x, idx_y]
            zx_ed = zx_st + self.config.train_params.ts_input_size 
            zy_ed = zy_st + self.config.train_params.ts_input_size

            # Handle the randomized noise input of the texture_synthesizer...
            outfeat_x_st = [start_pts_mesh[idx_x,idx_y,0] for start_pts_mesh in self.start_pts_mesh_outfeats]
            outfeat_y_st = [start_pts_mesh[idx_x,idx_y,1] for start_pts_mesh in self.start_pts_mesh_outfeats]
            outfeat_x_ed = [
                x_st + out_size for (x_st, out_size) in zip(outfeat_x_st, self.outfeat_sizes_list)]
            outfeat_y_ed = [
                y_st + out_size for (y_st, out_size) in zip(outfeat_y_st, self.outfeat_sizes_list)]
            noises = []
            for i, (fx_st, fy_st, fx_ed, fy_ed) in enumerate(zip(outfeat_x_st, outfeat_y_st, outfeat_x_ed, outfeat_y_ed)):
                # noises.append(testing_vars.noises[i][:, :, fx_st:fx_ed, fy_st:fy_ed].to(self.device))
                noises.append(self.circular_sample_width(
                    testing_vars.noises[i],
                    self.noise_widths[i],
                    fx_st,
                    fx_ed, 
                    fy_st,
                    fy_ed,
                    ).to(self.device))
            # 0:19, 0:17, 0:31, 0:29, 0:55, 0:53, 0:103, 0:101; x,y相同, 由start_pts_mesh控制
            # x相同, y为 12:31, 12:29, 24:55, 24:53, 48:103, 48:101, 96:199, 96:197  
            # noise 到 meta_height x meta_width, 和local_latent相对应, 这里偏置为12
            # local_latent的偏置为6
            # Deal with SS unfolding here
            zx_st -= self.ss_unfold_size
            zy_st -= self.ss_unfold_size
            zx_ed += self.ss_unfold_size
            zy_ed += self.ss_unfold_size
            
            # [Interactive] Decide whether the region will be updated, otherwise no need to generate
            if update_by_ss_map is not None:
                ss_cursors = zx_st, zx_ed, zy_st, zy_ed
                if not self.is_overlaping_update_map(update_by_ss_map, *ss_cursors):
                    continue
            if update_by_ts_map is not None:
                # For TS regional selection, we only select noises
                ts_cursors = outfeat_x_st[0], outfeat_x_ed[0], outfeat_y_st[0], outfeat_y_ed[0]
                if not self.is_overlaping_update_map(update_by_ts_map, *ts_cursors):
                    continue

            # cur_local_latent = testing_vars.local_latent[:, :, zx_st:zx_ed, zy_st:zy_ed].to(self.device)
            # # 0:35, 0:35; 下一循环为 0:35, 6:41
            # cur_coords = testing_vars.meta_coords[:, :, zx_st:zx_ed, zy_st:zy_ed].to(self.device)
            cur_local_latent = self.circular_sample_width(testing_vars.local_latent, testing_vars.local_latent.shape[3], zx_st, zx_ed, zy_st, zy_ed).to(self.device)
            cur_coords = self.circular_sample_width(testing_vars.meta_coords, testing_vars.meta_coords.shape[3], zx_st, zx_ed, zy_st, zy_ed).to(self.device)

            x_size = zx_ed - zx_st + 1
            y_size = zy_ed - zy_st + 1
            
            
            self.const_grid_size_x = testing_vars.meta_coords.shape[2]
            self.const_grid_size_y = testing_vars.meta_coords.shape[3]

            zy_st, circular_flag = self.get_circular_flag(zy_st, zy_ed, self.const_grid_size_y)

            if not self.attention_flag:
                print("\n\t\t\tATTENTION: Pre sampling mode is deactivate\n")
                self.attention_flag = True
            partial = 0.8
            if hasattr(self.config.train_params, "partial"):
                partial = self.config.train_params.partial
            coords_partial = {
                "p_x_st": zx_st / self.const_grid_size_x,
                "p_x_ed": (zx_st + x_size) / self.const_grid_size_x,
                "p_y_st": zy_st / self.const_grid_size_y,
                "p_y_ed": (zy_st + y_size) / self.const_grid_size_y,
                "circular_flag": circular_flag,
                "x_total": self.const_grid_size_x,
                "y_total": self.const_grid_size_y,
                "test_flag": True,
                "start_flag": iiter == 0,
                "h_step": zx_st // 6,
                "w_step": zy_st // 6,
                "y_st": zy_st,
                "y_ed": zy_ed,
                # "full_shape": self.full_shape,
                "partial": partial,
            }
            g_ema_kwargs = {
                "global_latent": testing_vars.global_latent,
                "local_latent": cur_local_latent,
                "override_coords": cur_coords,
                "coords_partial_override": coords_partial,
                "noises": noises,
                "disable_dual_latents": True,
                "calc_flops": calc_flops,
            }
            # 得到上面这些参数就可以生成小块
            """
            coords_partial = {
                    "p_x_st": x_st / self.const_grid_size_x,
                    "p_x_ed": (x_st + x_size) / self.const_grid_size_x,
                    "p_y_st": y_st / self.const_grid_size_y,
                    "p_y_ed": (y_st + y_size) / self.const_grid_size_y,
                    "circular_flag": not coords[0, 1, 1, 0] == torch.min(coords[0, 1, :, :]),
                    "x_total": self.const_grid_size_x,
                    "y_total": self.const_grid_size_y 
                }
            """
            if hasattr(testing_vars, "wplus_styles") and testing_vars.wplus_styles is not None:
                g_ema_kwargs["wplus_styles"] = testing_vars.wplus_styles
            img_x_st, img_y_st = outfeat_x_st[-1], outfeat_y_st[-1]
            img_x_ed, img_y_ed = outfeat_x_ed[-1], outfeat_y_ed[-1]
            index_tuple = (img_x_st, img_x_ed, img_y_st, img_y_ed) # 0-101, 0-101
            # 下一句生成小块
            exec_time, flops = self.maybe_parallel_inference(
                testing_vars, g_ema_kwargs=g_ema_kwargs, index_tuple=index_tuple, return_exec_time=write_gpu_time, calc_flops=calc_flops)
            accum_exec_time += exec_time
            if calc_flops:
                accum_flops_all += flops["all"]
                accum_flops_ss += flops["ss"]
                accum_flops_ts += flops["ts"]

        exec_time, flops = self.maybe_parallel_inference(
            testing_vars, flush=True, return_exec_time=write_gpu_time, calc_flops=calc_flops)
        if calc_flops:
            accum_flops_all += flops["all"]
            accum_flops_ss += flops["ss"]
            accum_flops_ts += flops["ts"]

        if write_gpu_time:
            accum_exec_time += exec_time
            print(" [*] GPU time {}".format(accum_exec_time))
            self.accum_exec_times.append(accum_exec_time)
            fmt_date = datetime.date.today().strftime("%d-%m-%Y")
            benchmark_file = os.path.join(self.save_root, "speed_benchmark_{}.txt".format(fmt_date))
            with open(benchmark_file, "a") as f:
                f.write("{:.6f}".format(accum_exec_time))

        if calc_flops:
            print(" [*] Total FLOPs: {} (SS {}, TS {})".format(
                self.pretty_print_flops(accum_flops_all), 
                self.pretty_print_flops(accum_flops_ss), 
                self.pretty_print_flops(accum_flops_ts)))

    def circular_sample_width(self, tensor:torch.Tensor, y_width, x_st, x_ed, y_st, y_ed):
        """_summary_

        Args:
            tensor (torch.Tensor): value to be sampled
            y_width (int): total_width of the tensor to be sampled
            y_st (int): start point of the slice (non-circular)
            y_ed (int): end point of the slice (non-circular: may be beyond the x_width)
        """
        if y_ed <= y_width:
            # 不超出循环的情况
            return tensor[:, :, x_st:x_ed, y_st:y_ed]
        elif y_ed <= y_width * 2:
            # 超出循环一倍以内的情况
            if y_st < y_width:
                y_ed = y_ed % y_width
                return torch.cat((tensor[:, :, x_st:x_ed, y_st:], tensor[:, :, x_st:x_ed, :y_ed]), dim=3)
            else:
                y_st = y_st % y_width
                y_ed = y_ed % y_width
                return tensor[:, :, x_st:x_ed, y_st:y_ed]
        else:
            # 超出循环两倍及以上进行递归求解
            # print(f"y_st is {y_st} and y_width is {y_width}")
            assert y_st >= y_width, "width should be larger than 35"
            y_st = y_st - y_width
            y_ed = y_ed - y_width
            return self.circular_sample_width(tensor, y_width, x_st, x_ed, y_st, y_ed)

    def save_results(self, meta_img, dump_vars=None):
        print(" [*] Saving results...")
        self.save_meta_imgs(meta_img)
        if dump_vars is not None:
            self.save_testing_vars(dump_vars)
        self.cur_global_id += self.config.train_params.batch_size

    def save_testing_vars(self, testing_vars):
        assert self.config.train_params.batch_size == 1, \
            "This is only designed to be used with the interactive tool."
        save_path = os.path.join(self.save_root, str(self.cur_global_id).zfill(6)+".pkl")
        pkl.dump(testing_vars, open(save_path, "wb"))

    def _wrap_feature(self, feat, wrap_size, dim):
        assert wrap_size < (feat.shape[dim] - 2*wrap_size), \
            "Does not expect the wrapping area is larger than a full period."
        if dim == 2:
            valid_st = feat[:, :, wrap_size:2*wrap_size]
            valid_ed = feat[:, :, -2*wrap_size:-wrap_size]
            feat[:, :, :wrap_size] = valid_ed
            feat[:, :, -wrap_size:] = valid_st
        elif dim == 3:
            valid_st = feat[:, :, :, wrap_size:2*wrap_size]
            valid_ed = feat[:, :, :, -2*wrap_size:-wrap_size]
            feat[:, :, :, :wrap_size] = valid_ed
            feat[:, :, :, -wrap_size:] = valid_st
        else:
            raise NotImplementedError(
                "I don't expect this function will be used other than spatial dims, but got {}.".format(dim))
        
    def save_meta_imgs(self, meta_img):
        self.full_image = meta_img
        # Center crop
        pad_h = (self.meta_height - self.target_height) // 2
        pad_w = (self.meta_width - self.target_width) // 2
        meta_img = meta_img[:, :, pad_h:pad_h+self.target_height, pad_w:pad_w+self.target_width]

        # Save the full image and the low-resolution image (for visualization)
        meta_img = meta_img.clamp(-1, 1).permute(0, 2, 3, 1)
        meta_img = (meta_img + 1) / 2
        meta_img_np = meta_img.numpy()
        before_id = self.cur_global_id
        for i in range(self.config.train_params.batch_size):
            global_id = self.cur_global_id + i
            save_path = os.path.join(self.save_root, str(global_id).zfill(6)+".png")
            plt.imsave(save_path, meta_img_np[i])
        after_id = global_id
        if self.save:
            save_path = os.path.join(self.latent_save_path, f"{before_id}-{after_id}.pth.tar")
            torch.save(self.save_dict, save_path)

    def _create_start_pts_mesh(self, step_size, num_steps_h, num_steps_w):
        # 定义重叠的尺寸
        start_pts_x = np.arange(num_steps_h) * step_size
        start_pts_y = np.arange(num_steps_w) * step_size
        start_pts_mesh = np.stack([
            np.repeat(start_pts_x.reshape(num_steps_h, 1), num_steps_w, axis=1),
            np.repeat(start_pts_y.reshape(1, num_steps_w), num_steps_h, axis=0),
        ], 2).astype(np.uint32) # shape: (H, W, 2)
        return start_pts_mesh

    def _init_starting_points(self):
        # 定义重叠的尺寸
        # Get all the starting points of the patches to generate, 计算生成的小块数量
        self.num_steps_h = \
            math.ceil((self.target_height - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad
        self.num_steps_w = \
            math.ceil((self.target_width  - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad

        self.start_pts_mesh_z = self._create_start_pts_mesh(
            step_size=self.latentspace_step_size, 
            num_steps_h=self.num_steps_h,
            num_steps_w=self.num_steps_w)
        self.start_pts_mesh_z += self.ss_unfold_size

        # Create this for:
        # (1) Final image pixels assignment
        # (2) Randomized noise handling within the texture synthesizer
        self.start_pts_mesh_outfeats = [
            self._create_start_pts_mesh(
                step_size=step_size,
                num_steps_h=self.num_steps_h,
                num_steps_w=self.num_steps_w,
            ) for step_size in self.outfeat_step_sizes]
        # start_pts_mesh_x = \
        #     (start_pts_mesh_z - ss_unfold_size) // latentspace_step_size * pixelspace_step_size # shape: (H, W, 2)
        # start_pts_mesh_x = start_pts_mesh_x.astype(np.uint32)

        # To avoid edge-condition on the image edge, we generate an image slightly larger than
        # requested, then center-crop to the requested resolution.
        self.meta_height = self.pixelspace_step_size * (self.num_steps_h-1) + self.outfeat_sizes_list[-1]
        self.meta_width  = self.pixelspace_step_size * (self.num_steps_w-1) + self.outfeat_sizes_list[-1]
        # height=485, width=773 (256 x 512)
        # height=485, width=965 (256 x 768)

    def compose_inv_root(self):
        return os.path.join("./logs/", self.config.var.exp_name, "test", self.config.task.prev_inv_config, "stats")

    def _init_close_loop_starting_points(self):
        # 定义重叠的尺寸
        # Get all the starting points of the patches to generate, 计算生成的小块数量
        self.num_steps_h = \
            math.ceil((self.target_height - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad
        # self.num_steps_w = \
        #     math.ceil((self.target_width  - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad
        assert self.target_width % self.pixelspace_step_size == 0, f"close-loop property is not ensured\n target width {self.target_width} is not divided by target patch size {self.pixelspace_step_size}"
        self.num_steps_w_min = math.ceil(self.target_width / self.pixelspace_step_size)
        # print(self.num_steps_w_min)

        if self.target_width == 1152:
            self.num_steps_w = self.num_steps_w_min + 2
            # for test
        else:
            self.num_steps_w = self.num_steps_w_min + 2
        # 多出的步数为重叠回来的步数
        # print(self.num_steps_w)
        
        self.start_pts_mesh_z = self._create_start_pts_mesh(
            step_size=self.latentspace_step_size, 
            num_steps_h=self.num_steps_h,
            num_steps_w=self.num_steps_w)
        self.start_pts_mesh_z += self.ss_unfold_size

        # Create this for:
        # (1) Final image pixels assignment
        # (2) Randomized noise handling within the texture synthesizer
        self.start_pts_mesh_outfeats = [
            self._create_start_pts_mesh(
                step_size=step_size,
                num_steps_h=self.num_steps_h,
                num_steps_w=self.num_steps_w,
            ) for step_size in self.outfeat_step_sizes]
        # start_pts_mesh_x = \
        #     (start_pts_mesh_z - ss_unfold_size) // latentspace_step_size * pixelspace_step_size # shape: (H, W, 2)
        # start_pts_mesh_x = start_pts_mesh_x.astype(np.uint32)

        # To avoid edge-condition on the image edge, we generate an image slightly larger than
        # requested, then center-crop to the requested resolution.
        self.meta_height = self.pixelspace_step_size * (self.num_steps_h-1) + self.outfeat_sizes_list[-1]
        self.meta_width  = self.num_steps_w_min * self.pixelspace_step_size
        # height=485, width=773 (256 x 512)
        # height=485, width=965 (256 x 768)

    def get_circular_flag(self, zy_st, zy_ed, y_total):
        # 训练时可以保证zy_st不超出坐标的取值范围, 测试时无法保证
        if zy_ed > y_total:
            if zy_st < y_total:
                circular_flag = True
                return zy_st, circular_flag
            else:
                circular_flag = False
                return zy_st % y_total, circular_flag
        else:
            circular_flag = False
            return zy_st, circular_flag


# patch coords + slicing 用于球卷积的生成器测试
class SlicingInfiniteGenerationManagerPatchCoordsCloseLoop(SlicingBaseTestManager):

    def save_full_imgs(self):
        # Save the full image and the low-resolution image (for visualization)
        meta_img = self.full_image
        meta_img = meta_img.clamp(-1, 1).permute(0, 2, 3, 1)
        meta_img = (meta_img + 1) / 2
        meta_img_np = meta_img.numpy()
        
        for i in range(self.config.train_params.batch_size):
            global_id = self.cur_global_id + i - self.config.train_params.batch_size
            # save after adding the global_id
            save_path = os.path.join(self.save_root, str(global_id).zfill(6)+"full.png")
            plt.imsave(save_path, meta_img_np[i])

    def task_specific_init(self, output_size=None):

        if output_size is None:
            self.target_height = self.config.task.height
            self.target_width = self.config.task.width
        else:
            self.target_height, self.target_width = output_size
           
        self._init_close_loop_starting_points()         
        self.noise_heights = self.outfeat_step_sizes * (self.num_steps_h-1) + self.outfeat_sizes_list
        self.noise_widths  = self.outfeat_step_sizes * self.num_steps_w_min

        if hasattr(self.config.task, "gen_from_inv_stats") and self.config.task.gen_from_inv_stats:
            self.inv_root = self.compose_inv_root()
            self.inv_rec_files = sorted(glob(os.path.join(self.inv_root, "*")))
            self.gen_from_inv_stats = True
        else:
            self.gen_from_inv_stats = False

    def run_next(self, save=True, write_gpu_time=False, inv_records=None, inv_placements=None, calc_flops=False, disable_pbar=False, **kwargs):
        if len(kwargs) > 0:
            for k,v in kwargs.items():
                if v is not None:
                    print(" [Warning] task manager receives untracked arg {} with value {}".format(k ,v))
        testing_vars = self.create_vars(inv_records=inv_records, inv_placements=inv_placements)
        self.generate(testing_vars, write_gpu_time=write_gpu_time, calc_flops=calc_flops, disable_pbar=disable_pbar)
        if save:
            self.save_results(testing_vars.meta_img)
        return testing_vars.meta_img

    def create_vars(self, inv_records=None, inv_placements=None):
        # 对local_latent进行采样
        mixing = False
        assert mixing == False, "Otherwise, an injection index must be specified and fed into g_ema."

        # Allocate memory for the final output, starts iterating and filling in the generated results.
        # Can be reused
        meta_img = torch.empty(
            self.config.train_params.batch_size,
            3,
            int(self.meta_height),
            int(self.meta_width)).float()

        # [Note]
        # 1.  One may implement a sophisticated version that does not required to 
        #     generate all the latents at once, as most of the info are not reusing
        #     during the inference. However, the author is just lazy and abusing his 
        #     CPU memory OuO
        global_latent = self.latent_sampler.sample_global_latent(
            self.config.train_params.batch_size, mixing=mixing, device=self.device)
        # full_local_latent_shape = (
        #     # Does not account GNN padding here, it is handled within the latent_sampler
        #     int(self.g_ema_module.calc_in_spatial_size(self.meta_height, include_ss=False)),
        #     int(self.g_ema_module.calc_in_spatial_size(self.meta_width, include_ss=False)),
        # )
        # local_latent = self.latent_sampler.sample_local_latent(
        #     self.config.train_params.batch_size, 
        #     device="cpu", # Store in CPU anyway, it can be INFINITLY LARGE!
        #     specific_shape=full_local_latent_shape)

        height = self.g_ema_module.calc_in_spatial_size(self.meta_height, include_ss=False) + self.padding_size
        if self.num_steps_w_min * 96 >= 35+self.padding_size:
            local_latent = self.latent_sampler.sample_circular_local_latent_patch101(
                self.config.train_params.batch_size,
                device="cpu",
                meta_width=self.meta_width,
                # padding_size=self.padding_size,
                height_in=height,
            )
        else:
            local_latent = self.latent_sampler.sample_circular_local_latent_patch101(
                self.config.train_params.batch_size,
                device="cpu",
                meta_width=self.meta_width,
                padding_size=self.padding_size,
                height_in=height,
            )

        meta_coords = self.coord_handler.sample_coord_grid(
            local_latent, 
            is_training=False) # 通过local_latent得到坐标

        # Create randomized noises, 得到随机噪声
        randomized_noises = [
            torch.randn(self.config.train_params.batch_size, 1, int(h), int(w))
                for (h,w) in zip(self.noise_heights, self.noise_widths)]

        testing_vars = TestingVars(
            meta_img=meta_img, 
            global_latent=global_latent, 
            local_latent=local_latent, 
            meta_coords=meta_coords, 
            noises=randomized_noises, 
            device=self.device)
        # 通过testing vars对测试的参数进行封装

        if self.gen_from_inv_stats:
            assert inv_records is None, \
                "`gen_from_inv_stats` already specified, should not receive `inv_records` from command!"
            assert self.config.train_params.batch_size == 1, \
                "Inverted parameters loading for batch is not yet implemeted! " + \
                "Please use parallel-batching instead, which provides a similar inference speed."
            inv_records = [self.inv_rec_files[self.cur_global_id]]
            inv_placements = [self.config.task.gen_from_inv_placement]

        if inv_records is not None:
            testing_vars.replace_by_records(
                self.g_ema_module, inv_records, inv_placements, assert_no_style=True)
        
        return testing_vars

    def generate(self, testing_vars, tkinter_pbar=None, update_by_ss_map=None, update_by_ts_map=None, 
                 write_gpu_time=False, calc_flops=False, disable_pbar=False):

        # I don't mind bruteforce casting combination here, cuz you should worry about the meta_img size first
        idx_tuples = list(iter_product(range(self.start_pts_mesh_z.shape[0]), range(self.start_pts_mesh_z.shape[1])))

        if disable_pbar:
            pbar = idx_tuples
        elif tkinter_pbar is not None:
            pbar = tkinter_pbar(idx_tuples)
        else:
            pbar = tqdm(idx_tuples)

        accum_exec_time = 0
        accum_flops_all, accum_flops_ss, accum_flops_ts = 0, 0, 0
        for iiter, (idx_x,idx_y) in enumerate(pbar):
            zx_st, zy_st = self.start_pts_mesh_z[idx_x, idx_y]
            zx_ed = zx_st + self.config.train_params.ts_input_size 
            zy_ed = zy_st + self.config.train_params.ts_input_size

            # Handle the randomized noise input of the texture_synthesizer...
            outfeat_x_st = [start_pts_mesh[idx_x,idx_y,0] for start_pts_mesh in self.start_pts_mesh_outfeats]
            outfeat_y_st = [start_pts_mesh[idx_x,idx_y,1] for start_pts_mesh in self.start_pts_mesh_outfeats]
            outfeat_x_ed = [
                x_st + out_size for (x_st, out_size) in zip(outfeat_x_st, self.outfeat_sizes_list)]
            outfeat_y_ed = [
                y_st + out_size for (y_st, out_size) in zip(outfeat_y_st, self.outfeat_sizes_list)]
            noises = []
            for i, (fx_st, fy_st, fx_ed, fy_ed) in enumerate(zip(outfeat_x_st, outfeat_y_st, outfeat_x_ed, outfeat_y_ed)):
                # noises.append(testing_vars.noises[i][:, :, fx_st:fx_ed, fy_st:fy_ed].to(self.device))
                noises.append(self.circular_sample_width(
                    testing_vars.noises[i],
                    self.noise_widths[i],
                    fx_st,
                    fx_ed, 
                    fy_st,
                    fy_ed,
                    ).to(self.device))
            # 0:19, 0:17, 0:31, 0:29, 0:55, 0:53, 0:103, 0:101; x,y相同, 由start_pts_mesh控制
            # x相同, y为 12:31, 12:29, 24:55, 24:53, 48:103, 48:101, 96:199, 96:197  
            # noise 到 meta_height x meta_width, 和local_latent相对应, 这里偏置为12
            # local_latent的偏置为6
            # Deal with SS unfolding here
            zx_st -= self.ss_unfold_size
            zy_st -= self.ss_unfold_size
            zx_ed += self.ss_unfold_size + self.padding_size
            zy_ed += self.ss_unfold_size + self.padding_size
            # 得到大块的zx_st, zy_st
            
            # [Interactive] Decide whether the region will be updated, otherwise no need to generate
            if update_by_ss_map is not None:
                ss_cursors = zx_st, zx_ed, zy_st, zy_ed
                if not self.is_overlaping_update_map(update_by_ss_map, *ss_cursors):
                    continue
            if update_by_ts_map is not None:
                # For TS regional selection, we only select noises
                ts_cursors = outfeat_x_st[0], outfeat_x_ed[0], outfeat_y_st[0], outfeat_y_ed[0]
                if not self.is_overlaping_update_map(update_by_ts_map, *ts_cursors):
                    continue

            # cur_local_latent = testing_vars.local_latent[:, :, zx_st:zx_ed, zy_st:zy_ed].to(self.device)
            # # 0:35, 0:35; 下一循环为 0:35, 6:41
            # cur_coords = testing_vars.meta_coords[:, :, zx_st:zx_ed, zy_st:zy_ed].to(self.device)
            cur_local_latent = self.circular_sample_width(testing_vars.local_latent, testing_vars.local_latent.shape[3], zx_st, zx_ed, zy_st, zy_ed).to(self.device)
            cur_coords = self.circular_sample_width(testing_vars.meta_coords, testing_vars.meta_coords.shape[3], zx_st, zx_ed, zy_st, zy_ed).to(self.device)

            x_size = zx_ed - zx_st + 1
            y_size = zy_ed - zy_st + 1
            
            
            self.const_grid_size_x = testing_vars.meta_coords.shape[2]
            self.const_grid_size_y = testing_vars.meta_coords.shape[3]

            zy_st, circular_flag = self.get_circular_flag(zy_st, zy_ed, self.const_grid_size_y)

            coords_partial = {
                "p_x_st": zx_st / self.const_grid_size_x,
                "p_x_ed": (zx_st + x_size) / self.const_grid_size_x,
                "p_y_st": zy_st / self.const_grid_size_y,
                "p_y_ed": (zy_st + y_size) / self.const_grid_size_y,
                "circular_flag": circular_flag,
                "x_total": self.const_grid_size_x,
                "y_total": self.const_grid_size_y,
                "test_flag": True,
                "start_flag": iiter == 0,
                "h_step": zx_st // 6,
                "w_step": zy_st // 6,
                "y_st": zy_st,
                "y_ed": zy_ed,
                "shrink": self.shrink_size,
                # "partial": 0.6,
            }
            g_ema_kwargs = {
                "global_latent": testing_vars.global_latent,
                "local_latent": cur_local_latent,
                "override_coords": cur_coords,
                "coords_partial_override": coords_partial,
                "noises": noises,
                "disable_dual_latents": True,
                "calc_flops": calc_flops,
            }
            # 得到上面这些参数就可以生成小块
            """
            coords_partial = {
                    "p_x_st": x_st / self.const_grid_size_x,
                    "p_x_ed": (x_st + x_size) / self.const_grid_size_x,
                    "p_y_st": y_st / self.const_grid_size_y,
                    "p_y_ed": (y_st + y_size) / self.const_grid_size_y,
                    "circular_flag": not coords[0, 1, 1, 0] == torch.min(coords[0, 1, :, :]),
                    "x_total": self.const_grid_size_x,
                    "y_total": self.const_grid_size_y 
                }
            """
            if hasattr(testing_vars, "wplus_styles") and testing_vars.wplus_styles is not None:
                g_ema_kwargs["wplus_styles"] = testing_vars.wplus_styles
            img_x_st, img_y_st = outfeat_x_st[-1], outfeat_y_st[-1]
            img_x_ed, img_y_ed = outfeat_x_ed[-1], outfeat_y_ed[-1]
            index_tuple = (img_x_st, img_x_ed, img_y_st, img_y_ed) # 0-101, 0-101
            # 下一句生成小块
            exec_time, flops = self.maybe_parallel_inference(
                testing_vars, g_ema_kwargs=g_ema_kwargs, index_tuple=index_tuple, return_exec_time=write_gpu_time, calc_flops=calc_flops)
            accum_exec_time += exec_time
            if calc_flops:
                accum_flops_all += flops["all"]
                accum_flops_ss += flops["ss"]
                accum_flops_ts += flops["ts"]

        exec_time, flops = self.maybe_parallel_inference(
            testing_vars, flush=True, return_exec_time=write_gpu_time, calc_flops=calc_flops)
        if calc_flops:
            accum_flops_all += flops["all"]
            accum_flops_ss += flops["ss"]
            accum_flops_ts += flops["ts"]
        
        if write_gpu_time:
            accum_exec_time += exec_time
            print(" [*] GPU time {}".format(accum_exec_time))
            self.accum_exec_times.append(accum_exec_time)
            fmt_date = datetime.date.today().strftime("%d-%m-%Y")
            benchmark_file = os.path.join(self.save_root, "speed_benchmark_{}.txt".format(fmt_date))
            with open(benchmark_file, "a") as f:
                f.write("{:.6f}".format(accum_exec_time))

        if calc_flops:
            print(" [*] Total FLOPs: {} (SS {}, TS {})".format(
                self.pretty_print_flops(accum_flops_all), 
                self.pretty_print_flops(accum_flops_ss), 
                self.pretty_print_flops(accum_flops_ts)))

    def circular_sample_width(self, tensor:torch.Tensor, y_width, x_st, x_ed, y_st, y_ed):
        """_summary_

        Args:
            tensor (torch.Tensor): value to be sampled
            y_width (int): total_width of the tensor to be sampled
            y_st (int): start point of the slice (non-circular)
            y_ed (int): end point of the slice (non-circular: may be beyond the x_width)
        """
        if y_ed <= y_width:
            # 不超出循环的情况
            return tensor[:, :, x_st:x_ed, y_st:y_ed]
        elif y_ed <= y_width * 2:
            # 超出循环一倍以内的情况
            if y_st < y_width:
                y_ed = y_ed % y_width
                return torch.cat((tensor[:, :, x_st:x_ed, y_st:], tensor[:, :, x_st:x_ed, :y_ed]), dim=3)
            else:
                y_st = y_st % y_width
                y_ed = y_ed % y_width
                return tensor[:, :, x_st:x_ed, y_st:y_ed]
        else:
            # 超出循环两倍及以上进行递归求解
            assert y_st > y_width
            y_st = y_st - y_width
            y_ed = y_ed - y_width
            return self.circular_sample_width(tensor, y_width, x_st, x_ed, y_st, y_ed)

    def save_results(self, meta_img, dump_vars=None):
        print(" [*] Saving results...")
        self.save_meta_imgs(meta_img)
        if dump_vars is not None:
            self.save_testing_vars(dump_vars)
        self.cur_global_id += self.config.train_params.batch_size

    def save_testing_vars(self, testing_vars):
        assert self.config.train_params.batch_size == 1, \
            "This is only designed to be used with the interactive tool."
        save_path = os.path.join(self.save_root, str(self.cur_global_id).zfill(6)+".pkl")
        pkl.dump(testing_vars, open(save_path, "wb"))

    def _wrap_feature(self, feat, wrap_size, dim):
        assert wrap_size < (feat.shape[dim] - 2*wrap_size), \
            "Does not expect the wrapping area is larger than a full period."
        if dim == 2:
            valid_st = feat[:, :, wrap_size:2*wrap_size]
            valid_ed = feat[:, :, -2*wrap_size:-wrap_size]
            feat[:, :, :wrap_size] = valid_ed
            feat[:, :, -wrap_size:] = valid_st
        elif dim == 3:
            valid_st = feat[:, :, :, wrap_size:2*wrap_size]
            valid_ed = feat[:, :, :, -2*wrap_size:-wrap_size]
            feat[:, :, :, :wrap_size] = valid_ed
            feat[:, :, :, -wrap_size:] = valid_st
        else:
            raise NotImplementedError(
                "I don't expect this function will be used other than spatial dims, but got {}.".format(dim))
        
    def save_meta_imgs(self, meta_img):
        self.full_image = meta_img
        # Center crop
        pad_h = (self.meta_height - self.target_height) // 2
        pad_w = (self.meta_width - self.target_width) // 2
        meta_img = meta_img[:, :, pad_h:pad_h+self.target_height, pad_w:pad_w+self.target_width]

        # Save the full image and the low-resolution image (for visualization)
        meta_img = meta_img.clamp(-1, 1).permute(0, 2, 3, 1)
        meta_img = (meta_img + 1) / 2
        meta_img_np = meta_img.numpy()
        
        for i in range(self.config.train_params.batch_size):
            global_id = self.cur_global_id + i
            save_path = os.path.join(self.save_root, str(global_id).zfill(6)+".png")
            plt.imsave(save_path, meta_img_np[i])

    def _create_start_pts_mesh(self, step_size, num_steps_h, num_steps_w):
        # 定义重叠的尺寸
        start_pts_x = np.arange(num_steps_h) * step_size
        start_pts_y = np.arange(num_steps_w) * step_size
        start_pts_mesh = np.stack([
            np.repeat(start_pts_x.reshape(num_steps_h, 1), num_steps_w, axis=1),
            np.repeat(start_pts_y.reshape(1, num_steps_w), num_steps_h, axis=0),
        ], 2).astype(np.uint32) # shape: (H, W, 2)
        return start_pts_mesh

    def _init_starting_points(self):
        # 定义重叠的尺寸
        # Get all the starting points of the patches to generate, 计算生成的小块数量
        self.num_steps_h = \
            math.ceil((self.target_height - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad
        self.num_steps_w = \
            math.ceil((self.target_width  - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad

        self.start_pts_mesh_z = self._create_start_pts_mesh(
            step_size=self.latentspace_step_size, 
            num_steps_h=self.num_steps_h,
            num_steps_w=self.num_steps_w)
        self.start_pts_mesh_z += self.ss_unfold_size

        # Create this for:
        # (1) Final image pixels assignment
        # (2) Randomized noise handling within the texture synthesizer
        self.start_pts_mesh_outfeats = [
            self._create_start_pts_mesh(
                step_size=step_size,
                num_steps_h=self.num_steps_h,
                num_steps_w=self.num_steps_w,
            ) for step_size in self.outfeat_step_sizes]
        # start_pts_mesh_x = \
        #     (start_pts_mesh_z - ss_unfold_size) // latentspace_step_size * pixelspace_step_size # shape: (H, W, 2)
        # start_pts_mesh_x = start_pts_mesh_x.astype(np.uint32)

        # To avoid edge-condition on the image edge, we generate an image slightly larger than
        # requested, then center-crop to the requested resolution.
        self.meta_height = self.pixelspace_step_size * (self.num_steps_h-1) + self.outfeat_sizes_list[-1]
        self.meta_width  = self.pixelspace_step_size * (self.num_steps_w-1) + self.outfeat_sizes_list[-1]
        # height=485, width=773 (256 x 512)
        # height=485, width=965 (256 x 768)

    def compose_inv_root(self):
        return os.path.join("./logs/", self.config.var.exp_name, "test", self.config.task.prev_inv_config, "stats")

    def _init_close_loop_starting_points(self):
        # 定义重叠的尺寸
        # Get all the starting points of the patches to generate, 计算生成的小块数量
        self.num_steps_h = \
            math.ceil((self.target_height - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad
        # self.num_steps_w = \
        #     math.ceil((self.target_width  - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad
        
        self.num_steps_w_min = math.ceil(self.target_width / self.pixelspace_step_size)
        self.num_steps_w = self.num_steps_w_min + 2
        # 多出的步数为重叠回来的步数
        print(self.num_steps_w)
        
        self.start_pts_mesh_z = self._create_start_pts_mesh(
            step_size=self.latentspace_step_size, 
            num_steps_h=self.num_steps_h,
            num_steps_w=self.num_steps_w)
        self.start_pts_mesh_z += self.ss_unfold_size

        # Create this for:
        # (1) Final image pixels assignment
        # (2) Randomized noise handling within the texture synthesizer
        self.start_pts_mesh_outfeats = [
            self._create_start_pts_mesh(
                step_size=step_size,
                num_steps_h=self.num_steps_h,
                num_steps_w=self.num_steps_w,
            ) for step_size in self.outfeat_step_sizes]
        # start_pts_mesh_x = \
        #     (start_pts_mesh_z - ss_unfold_size) // latentspace_step_size * pixelspace_step_size # shape: (H, W, 2)
        # start_pts_mesh_x = start_pts_mesh_x.astype(np.uint32)

        # To avoid edge-condition on the image edge, we generate an image slightly larger than
        # requested, then center-crop to the requested resolution.
        self.meta_height = self.pixelspace_step_size * (self.num_steps_h-1) + self.outfeat_sizes_list[-1]
        self.meta_width  = self.num_steps_w_min * self.pixelspace_step_size
        # height=485, width=773 (256 x 512)
        # height=485, width=965 (256 x 768)

    def get_circular_flag(self, zy_st, zy_ed, y_total):
        # 训练时可以保证zy_st不超出坐标的取值范围, 测试时无法保证
        if zy_ed > y_total:
            if zy_st < y_total:
                circular_flag = True
                return zy_st, circular_flag
            else:
                circular_flag = False
                return zy_st % y_total, circular_flag
        else:
            circular_flag = False
            return zy_st, circular_flag

    def get_inter_index(self, zy_st_big, zy_ed_big, width_big, width_small):
        # return zy_st_small, zy_ed_small
        # 
        pass


class InfiniteGenerationManagerPatchCoordsCloseLoopDatasetV2(BaseTestManager):
    attention_flag = False
    def save_full_imgs(self):
        # Save the full image and the low-resolution image (for visualization)
        meta_img = self.full_image
        meta_img = meta_img.clamp(-1, 1).permute(0, 2, 3, 1)
        meta_img = (meta_img + 1) / 2
        meta_img_np = meta_img.numpy()
        
        for i in range(self.config.train_params.batch_size):
            global_id = self.cur_global_id + i - self.config.train_params.batch_size
            # save after adding the global_id
            save_path = os.path.join(self.save_root, str(global_id).zfill(6)+"full.png")
            plt.imsave(save_path, meta_img_np[i])
            
    def task_specific_init(self, output_size=None):

        if output_size is None:
            self.target_height = self.config.task.height
            self.target_width = self.config.task.width
        else:
            self.target_height, self.target_width = output_size
           
        self._init_close_loop_starting_points()         
        self.noise_heights = self.outfeat_step_sizes * (self.num_steps_h-1) + self.outfeat_sizes_list
        self.noise_widths  = self.outfeat_step_sizes * self.num_steps_w_min

        if hasattr(self.config.task, "gen_from_inv_stats") and self.config.task.gen_from_inv_stats:
            self.inv_root = self.compose_inv_root()
            self.inv_rec_files = sorted(glob(os.path.join(self.inv_root, "*")))
            self.gen_from_inv_stats = True
        else:
            self.gen_from_inv_stats = False

    def run_next(self, save=True, write_gpu_time=False, inv_records=None, inv_placements=None, calc_flops=False, disable_pbar=False, **kwargs):
        if len(kwargs) > 0:
            for k,v in kwargs.items():
                if v is not None:
                    print(" [Warning] task manager receives untracked arg {} with value {}".format(k ,v))
        testing_vars = self.create_vars(inv_records=inv_records, inv_placements=inv_placements)
        self.generate(testing_vars, write_gpu_time=write_gpu_time, calc_flops=calc_flops, disable_pbar=disable_pbar)
        if save:
            self.save_results(testing_vars.meta_img)
        return testing_vars.meta_img

    def create_vars(self, inv_records=None, inv_placements=None):
        # 对local_latent进行采样
        mixing = False
        assert mixing == False, "Otherwise, an injection index must be specified and fed into g_ema."

        # Allocate memory for the final output, starts iterating and filling in the generated results.
        # Can be reused
        meta_img = torch.empty(
            self.config.train_params.batch_size,
            3,
            int(self.meta_height),
            int(self.meta_width)).float()

        # [Note]
        # 1.  One may implement a sophisticated version that does not required to 
        #     generate all the latents at once, as most of the info are not reusing
        #     during the inference. However, the author is just lazy and abusing his 
        #     CPU memory OuO
        global_latent = self.latent_sampler.sample_global_latent(
            self.config.train_params.batch_size, mixing=mixing, device=self.device)
        # full_local_latent_shape = (
        #     # Does not account GNN padding here, it is handled within the latent_sampler
        #     int(self.g_ema_module.calc_in_spatial_size(self.meta_height, include_ss=False)),
        #     int(self.g_ema_module.calc_in_spatial_size(self.meta_width, include_ss=False)),
        # )
        # local_latent = self.latent_sampler.sample_local_latent(
        #     self.config.train_params.batch_size, 
        #     device="cpu", # Store in CPU anyway, it can be INFINITLY LARGE!
        #     specific_shape=full_local_latent_shape)

        height = self.g_ema_module.calc_in_spatial_size(self.meta_height, include_ss=False)

        local_latent = self.latent_sampler.sample_circular_local_latent_patch101(
            self.config.train_params.batch_size,
            device="cpu",
            meta_width=self.meta_width,
            height_in=height,
        )

        self.full_shape = local_latent.shape[2:]

        meta_coords = self.coord_handler.sample_coord_grid(
            local_latent, 
            is_training=False) # 通过local_latent得到坐标

        # Create randomized noises, 得到随机噪声
        randomized_noises = [
            torch.randn(self.config.train_params.batch_size, 1, int(h), int(w))
                for (h,w) in zip(self.noise_heights, self.noise_widths)]

        testing_vars = TestingVars(
            meta_img=meta_img, 
            global_latent=global_latent, 
            local_latent=local_latent, 
            meta_coords=meta_coords, 
            noises=randomized_noises, 
            device=self.device)
        # 通过testing vars对测试的参数进行封装

        if self.gen_from_inv_stats:
            assert inv_records is None, \
                "`gen_from_inv_stats` already specified, should not receive `inv_records` from command!"
            assert self.config.train_params.batch_size == 1, \
                "Inverted parameters loading for batch is not yet implemeted! " + \
                "Please use parallel-batching instead, which provides a similar inference speed."
            inv_records = [self.inv_rec_files[self.cur_global_id]]
            inv_placements = [self.config.task.gen_from_inv_placement]

        if inv_records is not None:
            testing_vars.replace_by_records(
                self.g_ema_module, inv_records, inv_placements, assert_no_style=True)
        
        return testing_vars

    def generate(self, testing_vars, tkinter_pbar=None, update_by_ss_map=None, update_by_ts_map=None, 
                 write_gpu_time=False, calc_flops=False, disable_pbar=False):

        # I don't mind bruteforce casting combination here, cuz you should worry about the meta_img size first
        idx_tuples = list(iter_product(range(self.start_pts_mesh_z.shape[0]), range(self.start_pts_mesh_z.shape[1])))

        if disable_pbar:
            pbar = idx_tuples
        elif tkinter_pbar is not None:
            pbar = tkinter_pbar(idx_tuples)
        else:
            pbar = tqdm(idx_tuples)

        accum_exec_time = 0
        accum_flops_all, accum_flops_ss, accum_flops_ts = 0, 0, 0
        for iiter, (idx_x,idx_y) in enumerate(pbar):
            zx_st, zy_st = self.start_pts_mesh_z[idx_x, idx_y]
            zx_ed = zx_st + self.config.train_params.ts_input_size 
            zy_ed = zy_st + self.config.train_params.ts_input_size

            # Handle the randomized noise input of the texture_synthesizer...
            outfeat_x_st = [start_pts_mesh[idx_x,idx_y,0] for start_pts_mesh in self.start_pts_mesh_outfeats]
            outfeat_y_st = [start_pts_mesh[idx_x,idx_y,1] for start_pts_mesh in self.start_pts_mesh_outfeats]
            outfeat_x_ed = [
                x_st + out_size for (x_st, out_size) in zip(outfeat_x_st, self.outfeat_sizes_list)]
            outfeat_y_ed = [
                y_st + out_size for (y_st, out_size) in zip(outfeat_y_st, self.outfeat_sizes_list)]
            noises = []
            for i, (fx_st, fy_st, fx_ed, fy_ed) in enumerate(zip(outfeat_x_st, outfeat_y_st, outfeat_x_ed, outfeat_y_ed)):
                # noises.append(testing_vars.noises[i][:, :, fx_st:fx_ed, fy_st:fy_ed].to(self.device))
                noises.append(self.circular_sample_width(
                    testing_vars.noises[i],
                    self.noise_widths[i],
                    fx_st,
                    fx_ed, 
                    fy_st,
                    fy_ed,
                    ).to(self.device))
            # 0:19, 0:17, 0:31, 0:29, 0:55, 0:53, 0:103, 0:101; x,y相同, 由start_pts_mesh控制
            # x相同, y为 12:31, 12:29, 24:55, 24:53, 48:103, 48:101, 96:199, 96:197  
            # noise 到 meta_height x meta_width, 和local_latent相对应, 这里偏置为12
            # local_latent的偏置为6
            # Deal with SS unfolding here
            zx_st -= self.ss_unfold_size
            zy_st -= self.ss_unfold_size
            zx_ed += self.ss_unfold_size
            zy_ed += self.ss_unfold_size
            
            # [Interactive] Decide whether the region will be updated, otherwise no need to generate
            if update_by_ss_map is not None:
                ss_cursors = zx_st, zx_ed, zy_st, zy_ed
                if not self.is_overlaping_update_map(update_by_ss_map, *ss_cursors):
                    continue
            if update_by_ts_map is not None:
                # For TS regional selection, we only select noises
                ts_cursors = outfeat_x_st[0], outfeat_x_ed[0], outfeat_y_st[0], outfeat_y_ed[0]
                if not self.is_overlaping_update_map(update_by_ts_map, *ts_cursors):
                    continue

            # cur_local_latent = testing_vars.local_latent[:, :, zx_st:zx_ed, zy_st:zy_ed].to(self.device)
            # # 0:35, 0:35; 下一循环为 0:35, 6:41
            # cur_coords = testing_vars.meta_coords[:, :, zx_st:zx_ed, zy_st:zy_ed].to(self.device)
            cur_local_latent = self.circular_sample_width(testing_vars.local_latent, testing_vars.local_latent.shape[3], zx_st, zx_ed, zy_st, zy_ed).to(self.device)
            # NOTE: datasetV2 由partial_xy生成坐标格点而不直接传入坐标格点
            cur_coords = self.circular_sample_width(testing_vars.meta_coords, testing_vars.meta_coords.shape[3], zx_st, zx_ed, zy_st, zy_ed).to(self.device)

            x_size = zx_ed - zx_st + 1
            y_size = zy_ed - zy_st + 1
            
            
            self.const_grid_size_x = testing_vars.local_latent.shape[2]
            self.const_grid_size_y = testing_vars.local_latent.shape[3]

            zy_st, circular_flag = self.get_circular_flag(zy_st, zy_ed, self.const_grid_size_y)

            if not self.attention_flag:
                print("\n\t\t\tATTENTION: Pre sampling mode is deactivate\n")
                self.attention_flag = True

            coords_partial = {
                "p_x_st": zx_st / self.const_grid_size_x,
                "p_x_ed": (zx_st + x_size) / self.const_grid_size_x,
                "p_y_st": zy_st / self.const_grid_size_y,
                "p_y_ed": (zy_st + y_size) / self.const_grid_size_y,
                "circular_flag": circular_flag,
                "x_total": self.const_grid_size_x,
                "y_total": self.const_grid_size_y,
                "test_flag": True,
                "start_flag": iiter == 0,
                "h_step": zx_st // 6,
                "w_step": zy_st // 6,
                "y_st": zy_st,
                "y_ed": zy_ed,
                # "full_shape": self.full_shape,
                # "partial": 0.6,
            }
            # 得到partial_xy
            partial_xy = (
                coords_partial['p_x_st'],
                coords_partial["p_y_st"],
            )
            g_ema_kwargs = {
                "partial_xy": partial_xy,
                "global_latent": testing_vars.global_latent,
                "local_latent": cur_local_latent,
                "override_coords": cur_coords,
                "coords_partial_override": coords_partial,
                "noises": noises,
                "disable_dual_latents": True,
                "calc_flops": calc_flops,
            }
            # 得到上面这些参数就可以生成小块
            """
            coords_partial = {
                    "p_x_st": x_st / self.const_grid_size_x,
                    "p_x_ed": (x_st + x_size) / self.const_grid_size_x,
                    "p_y_st": y_st / self.const_grid_size_y,
                    "p_y_ed": (y_st + y_size) / self.const_grid_size_y,
                    "circular_flag": not coords[0, 1, 1, 0] == torch.min(coords[0, 1, :, :]),
                    "x_total": self.const_grid_size_x,
                    "y_total": self.const_grid_size_y 
                }
            """
            if hasattr(testing_vars, "wplus_styles") and testing_vars.wplus_styles is not None:
                g_ema_kwargs["wplus_styles"] = testing_vars.wplus_styles
            img_x_st, img_y_st = outfeat_x_st[-1], outfeat_y_st[-1]
            img_x_ed, img_y_ed = outfeat_x_ed[-1], outfeat_y_ed[-1]
            index_tuple = (img_x_st, img_x_ed, img_y_st, img_y_ed) # 0-101, 0-101
            # 下一句生成小块
            exec_time, flops = self.maybe_parallel_inference(
                testing_vars, g_ema_kwargs=g_ema_kwargs, index_tuple=index_tuple, return_exec_time=write_gpu_time, calc_flops=calc_flops)
            accum_exec_time += exec_time
            if calc_flops:
                accum_flops_all += flops["all"]
                accum_flops_ss += flops["ss"]
                accum_flops_ts += flops["ts"]

        # flash step
        exec_time, flops = self.maybe_parallel_inference(
            testing_vars, flush=True, return_exec_time=write_gpu_time, calc_flops=calc_flops)
        if calc_flops:
            accum_flops_all += flops["all"]
            accum_flops_ss += flops["ss"]
            accum_flops_ts += flops["ts"]

        if write_gpu_time:
            accum_exec_time += exec_time
            print(" [*] GPU time {}".format(accum_exec_time))
            self.accum_exec_times.append(accum_exec_time)
            fmt_date = datetime.date.today().strftime("%d-%m-%Y")
            benchmark_file = os.path.join(self.save_root, "speed_benchmark_{}.txt".format(fmt_date))
            with open(benchmark_file, "a") as f:
                f.write("{:.6f}".format(accum_exec_time))

        if calc_flops:
            print(" [*] Total FLOPs: {} (SS {}, TS {})".format(
                self.pretty_print_flops(accum_flops_all), 
                self.pretty_print_flops(accum_flops_ss), 
                self.pretty_print_flops(accum_flops_ts)))

    def circular_sample_width(self, tensor:torch.Tensor, y_width, x_st, x_ed, y_st, y_ed):
        """_summary_

        Args:
            tensor (torch.Tensor): value to be sampled
            y_width (int): total_width of the tensor to be sampled
            y_st (int): start point of the slice (non-circular)
            y_ed (int): end point of the slice (non-circular: may be beyond the x_width)
        """
        if y_ed <= y_width:
            # 不超出循环的情况
            return tensor[:, :, x_st:x_ed, y_st:y_ed]
        elif y_ed <= y_width * 2:
            # 超出循环一倍以内的情况
            if y_st < y_width:
                y_ed = y_ed % y_width
                return torch.cat((tensor[:, :, x_st:x_ed, y_st:], tensor[:, :, x_st:x_ed, :y_ed]), dim=3)
            else:
                y_st = y_st % y_width
                y_ed = y_ed % y_width
                return tensor[:, :, x_st:x_ed, y_st:y_ed]
        else:
            # 超出循环两倍及以上进行递归求解
            assert y_st > y_width
            y_st = y_st - y_width
            y_ed = y_ed - y_width
            return self.circular_sample_width(tensor, y_width, x_st, x_ed, y_st, y_ed)

    def save_results(self, meta_img, dump_vars=None):
        print(" [*] Saving results...")
        self.save_meta_imgs(meta_img)
        if dump_vars is not None:
            self.save_testing_vars(dump_vars)
        self.cur_global_id += self.config.train_params.batch_size

    def save_testing_vars(self, testing_vars):
        assert self.config.train_params.batch_size == 1, \
            "This is only designed to be used with the interactive tool."
        save_path = os.path.join(self.save_root, str(self.cur_global_id).zfill(6)+".pkl")
        pkl.dump(testing_vars, open(save_path, "wb"))

    def _wrap_feature(self, feat, wrap_size, dim):
        assert wrap_size < (feat.shape[dim] - 2*wrap_size), \
            "Does not expect the wrapping area is larger than a full period."
        if dim == 2:
            valid_st = feat[:, :, wrap_size:2*wrap_size]
            valid_ed = feat[:, :, -2*wrap_size:-wrap_size]
            feat[:, :, :wrap_size] = valid_ed
            feat[:, :, -wrap_size:] = valid_st
        elif dim == 3:
            valid_st = feat[:, :, :, wrap_size:2*wrap_size]
            valid_ed = feat[:, :, :, -2*wrap_size:-wrap_size]
            feat[:, :, :, :wrap_size] = valid_ed
            feat[:, :, :, -wrap_size:] = valid_st
        else:
            raise NotImplementedError(
                "I don't expect this function will be used other than spatial dims, but got {}.".format(dim))
        
    def save_meta_imgs(self, meta_img):
        self.full_image = meta_img
        # Center crop
        pad_h = (self.meta_height - self.target_height) // 2
        pad_w = (self.meta_width - self.target_width) // 2
        meta_img = meta_img[:, :, pad_h:pad_h+self.target_height, pad_w:pad_w+self.target_width]

        # Save the full image and the low-resolution image (for visualization)
        meta_img = meta_img.clamp(-1, 1).permute(0, 2, 3, 1)
        meta_img = (meta_img + 1) / 2
        meta_img_np = meta_img.numpy()
        
        for i in range(self.config.train_params.batch_size):
            global_id = self.cur_global_id + i
            save_path = os.path.join(self.save_root, str(global_id).zfill(6)+".png")
            plt.imsave(save_path, meta_img_np[i])

    def _create_start_pts_mesh(self, step_size, num_steps_h, num_steps_w):
        # 定义重叠的尺寸
        start_pts_x = np.arange(num_steps_h) * step_size
        start_pts_y = np.arange(num_steps_w) * step_size
        start_pts_mesh = np.stack([
            np.repeat(start_pts_x.reshape(num_steps_h, 1), num_steps_w, axis=1),
            np.repeat(start_pts_y.reshape(1, num_steps_w), num_steps_h, axis=0),
        ], 2).astype(np.uint32) # shape: (H, W, 2)
        return start_pts_mesh

    def _init_starting_points(self):
        # 定义重叠的尺寸
        # Get all the starting points of the patches to generate, 计算生成的小块数量
        self.num_steps_h = \
            math.ceil((self.target_height - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad
        self.num_steps_w = \
            math.ceil((self.target_width  - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad

        self.start_pts_mesh_z = self._create_start_pts_mesh(
            step_size=self.latentspace_step_size, 
            num_steps_h=self.num_steps_h,
            num_steps_w=self.num_steps_w)
        self.start_pts_mesh_z += self.ss_unfold_size

        # Create this for:
        # (1) Final image pixels assignment
        # (2) Randomized noise handling within the texture synthesizer
        self.start_pts_mesh_outfeats = [
            self._create_start_pts_mesh(
                step_size=step_size,
                num_steps_h=self.num_steps_h,
                num_steps_w=self.num_steps_w,
            ) for step_size in self.outfeat_step_sizes]
        # start_pts_mesh_x = \
        #     (start_pts_mesh_z - ss_unfold_size) // latentspace_step_size * pixelspace_step_size # shape: (H, W, 2)
        # start_pts_mesh_x = start_pts_mesh_x.astype(np.uint32)

        # To avoid edge-condition on the image edge, we generate an image slightly larger than
        # requested, then center-crop to the requested resolution.
        self.meta_height = self.pixelspace_step_size * (self.num_steps_h-1) + self.outfeat_sizes_list[-1]
        self.meta_width  = self.pixelspace_step_size * (self.num_steps_w-1) + self.outfeat_sizes_list[-1]
        # height=485, width=773 (256 x 512)
        # height=485, width=965 (256 x 768)

    def compose_inv_root(self):
        return os.path.join("./logs/", self.config.var.exp_name, "test", self.config.task.prev_inv_config, "stats")

    def _init_close_loop_starting_points(self):
        # 定义重叠的尺寸
        # Get all the starting points of the patches to generate, 计算生成的小块数量
        self.num_steps_h = \
            math.ceil((self.target_height - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad
        # self.num_steps_w = \
        #     math.ceil((self.target_width  - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad
        
        self.num_steps_w_min = math.ceil(self.target_width / self.pixelspace_step_size)
        self.num_steps_w = self.num_steps_w_min + 2
        # 多出的步数为重叠回来的步数
        
        self.start_pts_mesh_z = self._create_start_pts_mesh(
            step_size=self.latentspace_step_size, 
            num_steps_h=self.num_steps_h,
            num_steps_w=self.num_steps_w)
        self.start_pts_mesh_z += self.ss_unfold_size

        # Create this for:
        # (1) Final image pixels assignment
        # (2) Randomized noise handling within the texture synthesizer
        self.start_pts_mesh_outfeats = [
            self._create_start_pts_mesh(
                step_size=step_size,
                num_steps_h=self.num_steps_h,
                num_steps_w=self.num_steps_w,
            ) for step_size in self.outfeat_step_sizes]
        # start_pts_mesh_x = \
        #     (start_pts_mesh_z - ss_unfold_size) // latentspace_step_size * pixelspace_step_size # shape: (H, W, 2)
        # start_pts_mesh_x = start_pts_mesh_x.astype(np.uint32)

        # To avoid edge-condition on the image edge, we generate an image slightly larger than
        # requested, then center-crop to the requested resolution.
        self.meta_height = self.pixelspace_step_size * (self.num_steps_h-1) + self.outfeat_sizes_list[-1]
        self.meta_width  = self.num_steps_w_min * self.pixelspace_step_size
        # height=485, width=773 (256 x 512)
        # height=485, width=965 (256 x 768)

    def get_circular_flag(self, zy_st, zy_ed, y_total):
        # 训练时可以保证zy_st不超出坐标的取值范围, 测试时无法保证
        if zy_ed > y_total:
            if zy_st < y_total:
                circular_flag = True
                return zy_st, circular_flag
            else:
                circular_flag = False
                return zy_st % y_total, circular_flag
        else:
            circular_flag = False
            return zy_st, circular_flag


# TODO: 实现横坐标方向上以12进行采样的测试代码
# NOTE: 应用到baseline上
class InfiniteGenerationManagerCloseLoopNewInterval(BaseTestManagerNewInterval):
    
    if DEBUG_MODE:
        __logger__.info(f"class InfiniteGenerationManagerCloseLoopNewInterval is created")

    def task_specific_init(self, output_size=None):

        if output_size is None:
            self.target_height = self.config.task.height
            self.target_width = self.config.task.width
        else:
            self.target_height, self.target_width = output_size
           
        self._init_starting_points()         
        self.noise_heights = self.ori_outfeat_step_sizes * (self.num_steps_h-1) + self.outfeat_sizes_list
        self.noise_widths  = self.ori_outfeat_step_sizes * (self.num_steps_w_min)

        if hasattr(self.config.task, "gen_from_inv_stats") and self.config.task.gen_from_inv_stats:
            self.inv_root = self.compose_inv_root()
            self.inv_rec_files = sorted(glob(os.path.join(self.inv_root, "*")))
            self.gen_from_inv_stats = True
        else:
            self.gen_from_inv_stats = False

    def run_next(self, save=True, write_gpu_time=False, inv_records=None, inv_placements=None, calc_flops=False, disable_pbar=False, **kwargs):
        if len(kwargs) > 0:
            for k,v in kwargs.items():
                if v is not None:
                    print(" [Warning] task manager receives untracked arg {} with value {}".format(k ,v))
        testing_vars = self.create_vars(inv_records=inv_records, inv_placements=inv_placements)
        self.generate(testing_vars, write_gpu_time=write_gpu_time, calc_flops=calc_flops, disable_pbar=disable_pbar)
        if save:
            self.save_results(testing_vars.meta_img)
        return testing_vars.meta_img

    def create_vars(self, inv_records=None, inv_placements=None, seed=None):
        # 对local_latent进行采样
        mixing = False
        assert mixing == False, "Otherwise, an injection index must be specified and fed into g_ema."

        # Allocate memory for the final output, starts iterating and filling in the generated results.
        # Can be reused
        meta_img = torch.empty(
            self.config.train_params.batch_size,
            3,
            int(self.meta_height),
            int(self.meta_width)).float()

        # [Note]
        # 1.  One may implement a sophisticated version that does not required to 
        #     generate all the latents at once, as most of the info are not reusing
        #     during the inference. However, the author is just lazy and abusing his 
        #     CPU memory OuO
        global_latent = self.latent_sampler.sample_global_latent(
            self.config.train_params.batch_size, mixing=mixing, device=self.device, seed=seed)
        # full_local_latent_shape = (
        #     # Does not account GNN padding here, it is handled within the latent_sampler
        #     int(self.g_ema_module.calc_in_spatial_size(self.meta_height, include_ss=False)),
        #     int(self.g_ema_module.calc_in_spatial_size(self.meta_width, include_ss=False)),
        # )
        # local_latent = self.latent_sampler.sample_local_latent(
        #     self.config.train_params.batch_size, 
        #     device="cpu", # Store in CPU anyway, it can be INFINITLY LARGE!
        #     specific_shape=full_local_latent_shape)

        height = self.g_ema_module.calc_in_spatial_size(self.meta_height, include_ss=False)

        local_latent = self.latent_sampler.sample_circular_local_latent_patch101(
            self.config.train_params.batch_size,
            device="cpu",
            meta_width=self.meta_width,
            height_in=height,
            seed=seed,
            step_size=self.latentspace_step_size,
        )

        meta_coords = self.coord_handler.sample_coord_grid(
            local_latent, 
            is_training=False) # 通过local_latent得到坐标

        # Create randomized noises, 得到随机噪声
        # randomized_noises = [
        #     torch.randn(self.config.train_params.batch_size, 1, int(h), int(w))
        #         for (h,w) in zip(self.noise_heights, self.noise_widths)]
        
        randomized_noises = [
            torch.zeros(self.config.train_params.batch_size, 1, int(h), int(w))
                for (h,w) in zip(self.noise_heights, self.noise_widths)]

        testing_vars = TestingVars(
            meta_img=meta_img, 
            global_latent=global_latent, 
            local_latent=local_latent, 
            meta_coords=meta_coords, 
            noises=randomized_noises, 
            device=self.device)
        # 通过testing vars对测试的参数进行封装

        if self.gen_from_inv_stats:
            assert inv_records is None, \
                "`gen_from_inv_stats` already specified, should not receive `inv_records` from command!"
            assert self.config.train_params.batch_size == 1, \
                "Inverted parameters loading for batch is not yet implemeted! " + \
                "Please use parallel-batching instead, which provides a similar inference speed."
            inv_records = [self.inv_rec_files[self.cur_global_id]]
            inv_placements = [self.config.task.gen_from_inv_placement]

        if inv_records is not None:
            testing_vars.replace_by_records(
                self.g_ema_module, inv_records, inv_placements, assert_no_style=True)
        
        return testing_vars
    
    def circular_sample_width(self, tensor:torch.Tensor, y_width, x_st, x_ed, y_st, y_ed):
        """_summary_

        Args:
            tensor (torch.Tensor): value to be sampled
            y_width (int): total_width of the tensor to be sampled
            y_st (int): start point of the slice (non-circular)
            y_ed (int): end point of the slice (non-circular: may be beyond the x_width)
        """
        if y_ed <= y_width:
            # 不超出循环的情况
            return tensor[:, :, x_st:x_ed, y_st:y_ed]
        elif y_ed <= y_width * 2:
            # 超出循环一倍以内的情况
            if y_st < y_width:
                y_ed = y_ed % y_width
                return torch.cat((tensor[:, :, x_st:x_ed, y_st:], tensor[:, :, x_st:x_ed, :y_ed]), dim=3)
            else:
                y_st = y_st % y_width
                y_ed = y_ed % y_width
                return tensor[:, :, x_st:x_ed, y_st:y_ed]
        else:
            # 超出循环两倍及以上进行递归求解
            assert y_st > y_width
            y_st = y_st - y_width
            y_ed = y_ed - y_width
            return self.circular_sample_width(tensor, y_width, x_st, x_ed, y_st, y_ed)

    def generate(self, testing_vars, tkinter_pbar=None, update_by_ss_map=None, update_by_ts_map=None, 
                 write_gpu_time=False, calc_flops=False, disable_pbar=False):

        # I don't mind bruteforce casting combination here, cuz you should worry about the meta_img size first
        idx_tuples = list(iter_product(range(self.start_pts_mesh_z.shape[0]), range(self.start_pts_mesh_z.shape[1])))

        if disable_pbar:
            pbar = idx_tuples
        elif tkinter_pbar is not None:
            pbar = tkinter_pbar(idx_tuples)
        else:
            pbar = tqdm(idx_tuples)

        accum_exec_time = 0
        accum_flops_all, accum_flops_ss, accum_flops_ts = 0, 0, 0
        for iiter, (idx_x,idx_y) in enumerate(pbar):
            zx_st, zy_st = self.start_pts_mesh_z[idx_x, idx_y]
            zx_ed = zx_st + self.config.train_params.ts_input_size 
            zy_ed = zy_st + self.config.train_params.ts_input_size

            # Handle the randomized noise input of the texture_synthesizer...
            outfeat_x_st = [start_pts_mesh[idx_x,idx_y,0] for start_pts_mesh in self.start_pts_mesh_outfeats]
            outfeat_y_st = [start_pts_mesh[idx_x,idx_y,1] for start_pts_mesh in self.start_pts_mesh_outfeats]
            output_img_y_st = [start_pts_mesh[idx_x,idx_y,1] for start_pts_mesh in self.ori_start_pts_mesh_outfeats]
            outfeat_x_ed = [
                x_st + out_size for (x_st, out_size) in zip(outfeat_x_st, self.outfeat_sizes_list)]
            outfeat_y_ed = [
                y_st + out_size for (y_st, out_size) in zip(outfeat_y_st, self.outfeat_sizes_list)]
            output_img_y_ed = [
                y_st + out_size for (y_st, out_size) in zip(output_img_y_st, self.outfeat_sizes_list)
            ]
            noises = []
            
            if DEBUG_MODE:
                __logger__.info(
                    f"iter {iiter}:"
                    f"x_st is {outfeat_x_st}\n"
                    f"x_ed is {outfeat_x_ed}\n"
                    f"y_st is {outfeat_y_st}\n"
                    f"y_ed is {outfeat_y_ed}\n"
                    )

            for i, (fx_st, fy_st, fx_ed, fy_ed) in enumerate(zip(outfeat_x_st, output_img_y_st, outfeat_x_ed, output_img_y_ed)):
                # 实现循环采样
                noises.append(self.circular_sample_width(testing_vars.noises[i], self.noise_widths[i], fx_st, fx_ed, fy_st, fy_ed).to(self.device))
            # 0:19, 0:17, 0:31, 0:29, 0:55, 0:53, 0:103, 0:101; x,y相同, 由start_pts_mesh控制
            # x相同, y为 12:31, 12:29, 24:55, 24:53, 48:103, 48:101, 96:199, 96:197  
            # noise 到 meta_height x meta_width, 和local_latent相对应, 这里偏置为12
            # local_latent的偏置为6 
            # Deal with SS unfolding here
            zx_st -= self.ss_unfold_size
            zy_st -= self.ss_unfold_size
            zx_ed += self.ss_unfold_size
            zy_ed += self.ss_unfold_size
            
            # [Interactive] Decide whether the region will be updated, otherwise no need to generate
            if update_by_ss_map is not None:
                ss_cursors = zx_st, zx_ed, zy_st, zy_ed
                if not self.is_overlaping_update_map(update_by_ss_map, *ss_cursors):
                    continue
            if update_by_ts_map is not None:
                # For TS regional selection, we only select noises
                ts_cursors = outfeat_x_st[0], outfeat_x_ed[0], outfeat_y_st[0], outfeat_y_ed[0]
                if not self.is_overlaping_update_map(update_by_ts_map, *ts_cursors):
                    continue
            
            # 实现循环采样
            cur_local_latent = self.circular_sample_width(testing_vars.local_latent, testing_vars.local_latent.shape[3], zx_st, zx_ed, zy_st, zy_ed).to(self.device)
            # 0:35, 0:35; 下一循环为 0:35, 6:41

            # 实现循环采样
            cur_coords = self.circular_sample_width(testing_vars.meta_coords, testing_vars.meta_coords.shape[3], zx_st, zx_ed, zy_st, zy_ed).to(self.device)
            g_ema_kwargs = {
                "global_latent": testing_vars.global_latent,
                "local_latent": cur_local_latent,
                "override_coords": cur_coords,
                "noises": noises,
                "disable_dual_latents": True,
                "calc_flops": calc_flops,
            }
            # 得到上面这些参数就可以生成小块
            if hasattr(testing_vars, "wplus_styles") and testing_vars.wplus_styles is not None:
                g_ema_kwargs["wplus_styles"] = testing_vars.wplus_styles
            img_x_st, img_y_st = outfeat_x_st[-1], output_img_y_st[-1]
            img_x_ed, img_y_ed = outfeat_x_ed[-1], output_img_y_ed[-1]
            index_tuple = (img_x_st, img_x_ed, img_y_st, img_y_ed) # 0-101, 0-101
            # 下一句生成小块
            exec_time, flops = self.maybe_parallel_inference(
                testing_vars, g_ema_kwargs=g_ema_kwargs, index_tuple=index_tuple, return_exec_time=write_gpu_time, calc_flops=calc_flops)
            accum_exec_time += exec_time
            if calc_flops:
                accum_flops_all += flops["all"]
                accum_flops_ss += flops["ss"]
                accum_flops_ts += flops["ts"]

        exec_time, flops = self.maybe_parallel_inference(
            testing_vars, flush=True, return_exec_time=write_gpu_time, calc_flops=calc_flops)
        if calc_flops:
            accum_flops_all += flops["all"]
            accum_flops_ss += flops["ss"]
            accum_flops_ts += flops["ts"]
        
        if write_gpu_time:
            accum_exec_time += exec_time
            print(" [*] GPU time {}".format(accum_exec_time))
            self.accum_exec_times.append(accum_exec_time)
            fmt_date = datetime.date.today().strftime("%d-%m-%Y")
            benchmark_file = os.path.join(self.save_root, "speed_benchmark_{}.txt".format(fmt_date))
            with open(benchmark_file, "a") as f:
                f.write("{:.6f}".format(accum_exec_time))

        if calc_flops:
            print(" [*] Total FLOPs: {} (SS {}, TS {})".format(
                self.pretty_print_flops(accum_flops_all), 
                self.pretty_print_flops(accum_flops_ss), 
                self.pretty_print_flops(accum_flops_ts)))

    def save_results(self, meta_img, dump_vars=None):
        print(" [*] Saving results...")
        self.save_meta_imgs(meta_img)
        if dump_vars is not None:
            self.save_testing_vars(dump_vars)
        self.cur_global_id += self.config.train_params.batch_size

    def save_testing_vars(self, testing_vars):
        assert self.config.train_params.batch_size == 1, \
            "This is only designed to be used with the interactive tool."
        save_path = os.path.join(self.save_root, str(self.cur_global_id).zfill(6)+".pkl")
        pkl.dump(testing_vars, open(save_path, "wb"))

    def _wrap_feature(self, feat, wrap_size, dim):
        assert wrap_size < (feat.shape[dim] - 2*wrap_size), \
            "Does not expect the wrapping area is larger than a full period."
        if dim == 2:
            valid_st = feat[:, :, wrap_size:2*wrap_size]
            valid_ed = feat[:, :, -2*wrap_size:-wrap_size]
            feat[:, :, :wrap_size] = valid_ed
            feat[:, :, -wrap_size:] = valid_st
        elif dim == 3:
            valid_st = feat[:, :, :, wrap_size:2*wrap_size]
            valid_ed = feat[:, :, :, -2*wrap_size:-wrap_size]
            feat[:, :, :, :wrap_size] = valid_ed
            feat[:, :, :, -wrap_size:] = valid_st
        else:
            raise NotImplementedError(
                "I don't expect this function will be used other than spatial dims, but got {}.".format(dim))
        
    def save_meta_imgs(self, meta_img):
        self.full_image = meta_img
        # Center crop
        pad_h = (self.meta_height - self.target_height) // 2
        pad_w = (self.meta_width - self.target_width) // 2
        meta_img = meta_img[:, :, pad_h:pad_h+self.target_height, pad_w:pad_w+self.target_width]

        # Save the full image and the low-resolution image (for visualization)
        meta_img = meta_img.clamp(-1, 1).permute(0, 2, 3, 1)
        meta_img = (meta_img + 1) / 2
        meta_img_np = meta_img.numpy()
        
        for i in range(self.config.train_params.batch_size):
            global_id = self.cur_global_id + i
            save_path = os.path.join(self.save_root, str(global_id).zfill(6)+".png")
            plt.imsave(save_path, meta_img_np[i])

    def save_full_imgs(self):
        # Save the full image and the low-resolution image (for visualization)
        meta_img = self.full_image
        meta_img = meta_img.clamp(-1, 1).permute(0, 2, 3, 1)
        meta_img = (meta_img + 1) / 2
        meta_img_np = meta_img.numpy()
        
        for i in range(self.config.train_params.batch_size):
            global_id = self.cur_global_id + i - self.config.train_params.batch_size
            # save after adding the global_id
            save_path = os.path.join(self.save_root, str(global_id).zfill(6)+"full.png")
            plt.imsave(save_path, meta_img_np[i])

    def _create_start_pts_mesh(self, step_size, ori_step_size, num_steps_h, num_steps_w):
        # 定义重叠的尺寸
        # step_size: w
        # ori_step_size: h
        start_pts_x = np.arange(num_steps_h) * ori_step_size
        start_pts_y = np.arange(num_steps_w) * step_size
        start_pts_mesh = np.stack([
            np.repeat(start_pts_x.reshape(num_steps_h, 1), num_steps_w, axis=1),
            np.repeat(start_pts_y.reshape(1, num_steps_w), num_steps_h, axis=0),
        ], 2).astype(np.uint32) # shape: (H, W, 2)
        return start_pts_mesh

    def _init_starting_points(self):
        # 定义重叠的尺寸
        # Get all the starting points of the patches to generate, 计算生成的小块数量
        self.num_steps_h = \
            math.ceil((self.target_height - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad
        # self.num_steps_w = \
        #     math.ceil((self.target_width  - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad
        
        self.num_steps_w_min = math.ceil(self.target_width / self.pixelspace_step_size)
        self.num_steps_w = self.num_steps_w_min + 1
        # 多出的步数为重叠回来的步数
        
        self.start_pts_mesh_z = self._create_start_pts_mesh(
            step_size=self.latentspace_step_size,
            ori_step_size=self.ori_latentspace_step_size, 
            num_steps_h=self.num_steps_h,
            num_steps_w=self.num_steps_w
            )
        self.start_pts_mesh_z += self.ss_unfold_size

        # Create this for:
        # (1) Final image pixels assignment
        # (2) Randomized noise handling within the texture synthesizer
        self.start_pts_mesh_outfeats = [
            self._create_start_pts_mesh(
                step_size=step_size,
                ori_step_size=ori_step_size,
                num_steps_h=self.num_steps_h,
                num_steps_w=self.num_steps_w,
            ) for ori_step_size, step_size in zip(self.ori_outfeat_step_sizes, self.outfeat_step_sizes)]


        self.ori_start_pts_mesh_outfeats = [
            self._create_start_pts_mesh(
                step_size=ori_step_size,
                ori_step_size=ori_step_size,
                num_steps_h=self.num_steps_h,
                num_steps_w=self.num_steps_w,
            ) for ori_step_size in self.ori_outfeat_step_sizes]
        # start_pts_mesh_x = \
        #     (start_pts_mesh_z - ss_unfold_size) // latentspace_step_size * pixelspace_step_size # shape: (H, W, 2)
        # start_pts_mesh_x = start_pts_mesh_x.astype(np.uint32)

        # To avoid edge-condition on the image edge, we generate an image slightly larger than
        # requested, then center-crop to the requested resolution.
        self.meta_height = self.pixelspace_step_size * (self.num_steps_h-1) + self.outfeat_sizes_list[-1]
        self.meta_width  = self.num_steps_w_min * self.pixelspace_step_size
        # height=485, width=773 (256 x 512)
        # height=485, width=965 (256 x 768)

    def compose_inv_root(self):
        return os.path.join("./logs/", self.config.var.exp_name, "test", self.config.task.prev_inv_config, "stats")


# 隐变量插值实验
class InfiniteGenerationManagerPatchCoordsCloseLoopInterpolate(BaseTestManager):
    attention_flag = False
    def save_full_imgs(self):
        # Save the full image and the low-resolution image (for visualization)
        meta_img = self.full_image
        meta_img = meta_img.clamp(-1, 1).permute(0, 2, 3, 1)
        meta_img = (meta_img + 1) / 2
        meta_img_np = meta_img.numpy()
        
        for i in range(self.config.train_params.batch_size):
            global_id = self.cur_global_id + i - self.config.train_params.batch_size
            # save after adding the global_id
            save_path = os.path.join(self.save_root, str(global_id).zfill(6)+"full.png")
            plt.imsave(save_path, meta_img_np[i])
            
    def task_specific_init(self, output_size=None, **kwargs):

        if output_size is None:
            self.target_height = self.config.task.height
            self.target_width = self.config.task.width
        else:
            self.target_height, self.target_width = output_size
           
        self._init_close_loop_starting_points()         
        self.noise_heights = self.outfeat_step_sizes * (self.num_steps_h-1) + self.outfeat_sizes_list
        self.noise_widths  = self.outfeat_step_sizes * self.num_steps_w_min

        if hasattr(self.config.task, "gen_from_inv_stats") and self.config.task.gen_from_inv_stats:
            self.inv_root = self.compose_inv_root()
            self.inv_rec_files = sorted(glob(os.path.join(self.inv_root, "*")))
            self.gen_from_inv_stats = True
        else:
            self.gen_from_inv_stats = False

        self._sample_inter(seed=kwargs.get("seeds", None))

    def _sample_inter(self, seed):
        self.global_1 = self.latent_sampler.sample_global_latent(
            self.config.train_params.batch_size, mixing=False, device=self.device, seed=seed)

        self.global_2 = self.latent_sampler.sample_global_latent(
            self.config.train_params.batch_size, mixing=False, device=self.device, seed=seed)

        height = self.g_ema_module.calc_in_spatial_size(self.meta_height, include_ss=False)

        self.local_1 = self.latent_sampler.sample_circular_local_latent_patch101(
            self.config.train_params.batch_size,
            device="cpu",
            meta_width=self.meta_width,
            height_in=height,
            seed=seed,
        )
        self.local_2 = self.latent_sampler.sample_circular_local_latent_patch101(
            self.config.train_params.batch_size,
            device="cpu",
            meta_width=self.meta_width,
            height_in=height,
            seed=seed,
        )

        self.r_noise_1 = [
            torch.randn(self.config.train_params.batch_size, 1, int(h), int(w))
                for (h,w) in zip(self.noise_heights, self.noise_widths)]

        self.r_noise_2 = [
            torch.randn(self.config.train_params.batch_size, 1, int(h), int(w))
                for (h,w) in zip(self.noise_heights, self.noise_widths)]
    
    def run_next(self, save=True, write_gpu_time=False, inv_records=None, inv_placements=None, calc_flops=False, disable_pbar=False, **kwargs):
        if len(kwargs) > 0:
            for k,v in kwargs.items():
                if v is not None and k not in ["seeds"]:
                    print(" [Warning] task manager receives untracked arg {} with value {}".format(k ,v))
        testing_vars = self.create_vars(inv_records=inv_records, inv_placements=inv_placements, seed=kwargs.get("seeds", None), _iter=kwargs.get("_iter", None))
        self.generate(testing_vars, write_gpu_time=write_gpu_time, calc_flops=calc_flops, disable_pbar=disable_pbar)
        if save:
            self.save_results(testing_vars.meta_img)
        return testing_vars.meta_img

    def create_vars(self, inv_records=None, inv_placements=None, seed=None, _iter=None):
        # 对local_latent进行采样
        assert _iter is not None, "Please specify the iteration number for the interpolation."
        mixing = False
        assert mixing == False, "Otherwise, an injection index must be specified and fed into g_ema."
        
        # Allocate memory for the final output, starts iterating and filling in the generated results.
        # Can be reused
        meta_img = torch.empty(
            self.config.train_params.batch_size,
            3,
            int(self.meta_height),
            int(self.meta_width)).float()

        # [Note]
        # 1.  One may implement a sophisticated version that does not required to 
        #     generate all the latents at once, as most of the info are not reusing
        #     during the inference. However, the author is just lazy and abusing his 
        #     CPU memory OuO
        # global_latent = self.latent_sampler.sample_global_latent(
        #     self.config.train_params.batch_size, mixing=mixing, device=self.device, seed=seed)
        # full_local_latent_shape = (
        #     # Does not account GNN padding here, it is handled within the latent_sampler
        #     int(self.g_ema_module.calc_in_spatial_size(self.meta_height, include_ss=False)),
        #     int(self.g_ema_module.calc_in_spatial_size(self.meta_width, include_ss=False)),
        # )
        # local_latent = self.latent_sampler.sample_local_latent(
        #     self.config.train_params.batch_size, 
        #     device="cpu", # Store in CPU anyway, it can be INFINITLY LARGE!
        #     specific_shape=full_local_latent_shape)

        # height = self.g_ema_module.calc_in_spatial_size(self.meta_height, include_ss=False)

        # local_latent = self.latent_sampler.sample_circular_local_latent_patch101(
        #     self.config.train_params.batch_size,
        #     device="cpu",
        #     meta_width=self.meta_width,
        #     height_in=height,
        #     seed=seed,
        # )
        alpha = np.linspace(0, 1, self.config.task.num_gen)[_iter]
        global_latent = (1 - alpha) * self.global_1 + alpha * self.global_2
        local_latent = (1 - alpha) * self.local_1 + alpha * self.local_2

        self.full_shape = local_latent.shape[2:]

        meta_coords = self.coord_handler.sample_coord_grid(
            local_latent, 
            is_training=False) # 通过local_latent得到坐标

        # Create randomized noises, 得到随机噪声
        # 噪声也由seed控制，保证一样
        # if seed is not None:
        #     randomized_noises = [
        #         torch.from_numpy(np.random.RandomState(seed).randn(self.config.train_params.batch_size, 1, int(h), int(w)))
        #             for (h,w) in zip(self.noise_heights, self.noise_widths)]
        # else:
        randomized_noises = [(1 - alpha) * e1 + alpha * e2 for (e1,e2) in zip(self.r_noise_1, self.r_noise_2)]

        testing_vars = TestingVars(
            meta_img=meta_img, 
            global_latent=global_latent, 
            local_latent=local_latent, 
            meta_coords=meta_coords, 
            noises=randomized_noises, 
            device=self.device)
        # 通过testing vars对测试的参数进行封装

        if self.gen_from_inv_stats:
            assert inv_records is None, \
                "`gen_from_inv_stats` already specified, should not receive `inv_records` from command!"
            assert self.config.train_params.batch_size == 1, \
                "Inverted parameters loading for batch is not yet implemeted! " + \
                "Please use parallel-batching instead, which provides a similar inference speed."
            inv_records = [self.inv_rec_files[self.cur_global_id]]
            inv_placements = [self.config.task.gen_from_inv_placement]

        if inv_records is not None:
            testing_vars.replace_by_records(
                self.g_ema_module, inv_records, inv_placements, assert_no_style=True)
        
        return testing_vars

    def generate(self, testing_vars, tkinter_pbar=None, update_by_ss_map=None, update_by_ts_map=None, 
                 write_gpu_time=False, calc_flops=False, disable_pbar=False):

        # I don't mind bruteforce casting combination here, cuz you should worry about the meta_img size first
        idx_tuples = list(iter_product(range(self.start_pts_mesh_z.shape[0]), range(self.start_pts_mesh_z.shape[1])))

        if disable_pbar:
            pbar = idx_tuples
        elif tkinter_pbar is not None:
            pbar = tkinter_pbar(idx_tuples)
        else:
            pbar = tqdm(idx_tuples)

        accum_exec_time = 0
        accum_flops_all, accum_flops_ss, accum_flops_ts = 0, 0, 0
        for iiter, (idx_x,idx_y) in enumerate(pbar):
            zx_st, zy_st = self.start_pts_mesh_z[idx_x, idx_y]
            zx_ed = zx_st + self.config.train_params.ts_input_size 
            zy_ed = zy_st + self.config.train_params.ts_input_size

            # Handle the randomized noise input of the texture_synthesizer...
            outfeat_x_st = [start_pts_mesh[idx_x,idx_y,0] for start_pts_mesh in self.start_pts_mesh_outfeats]
            outfeat_y_st = [start_pts_mesh[idx_x,idx_y,1] for start_pts_mesh in self.start_pts_mesh_outfeats]
            outfeat_x_ed = [
                x_st + out_size for (x_st, out_size) in zip(outfeat_x_st, self.outfeat_sizes_list)]
            outfeat_y_ed = [
                y_st + out_size for (y_st, out_size) in zip(outfeat_y_st, self.outfeat_sizes_list)]
            noises = []
            for i, (fx_st, fy_st, fx_ed, fy_ed) in enumerate(zip(outfeat_x_st, outfeat_y_st, outfeat_x_ed, outfeat_y_ed)):
                # noises.append(testing_vars.noises[i][:, :, fx_st:fx_ed, fy_st:fy_ed].to(self.device))
                noises.append(self.circular_sample_width(
                    testing_vars.noises[i],
                    self.noise_widths[i],
                    fx_st,
                    fx_ed, 
                    fy_st,
                    fy_ed,
                    ).to(self.device))
            # 0:19, 0:17, 0:31, 0:29, 0:55, 0:53, 0:103, 0:101; x,y相同, 由start_pts_mesh控制
            # x相同, y为 12:31, 12:29, 24:55, 24:53, 48:103, 48:101, 96:199, 96:197  
            # noise 到 meta_height x meta_width, 和local_latent相对应, 这里偏置为12
            # local_latent的偏置为6
            # Deal with SS unfolding here
            zx_st -= self.ss_unfold_size
            zy_st -= self.ss_unfold_size
            zx_ed += self.ss_unfold_size
            zy_ed += self.ss_unfold_size
            
            # [Interactive] Decide whether the region will be updated, otherwise no need to generate
            if update_by_ss_map is not None:
                ss_cursors = zx_st, zx_ed, zy_st, zy_ed
                if not self.is_overlaping_update_map(update_by_ss_map, *ss_cursors):
                    continue
            if update_by_ts_map is not None:
                # For TS regional selection, we only select noises
                ts_cursors = outfeat_x_st[0], outfeat_x_ed[0], outfeat_y_st[0], outfeat_y_ed[0]
                if not self.is_overlaping_update_map(update_by_ts_map, *ts_cursors):
                    continue

            # cur_local_latent = testing_vars.local_latent[:, :, zx_st:zx_ed, zy_st:zy_ed].to(self.device)
            # # 0:35, 0:35; 下一循环为 0:35, 6:41
            # cur_coords = testing_vars.meta_coords[:, :, zx_st:zx_ed, zy_st:zy_ed].to(self.device)
            cur_local_latent = self.circular_sample_width(testing_vars.local_latent, testing_vars.local_latent.shape[3], zx_st, zx_ed, zy_st, zy_ed).to(self.device)
            cur_coords = self.circular_sample_width(testing_vars.meta_coords, testing_vars.meta_coords.shape[3], zx_st, zx_ed, zy_st, zy_ed).to(self.device)

            x_size = zx_ed - zx_st + 1
            y_size = zy_ed - zy_st + 1
            
            
            self.const_grid_size_x = testing_vars.meta_coords.shape[2]
            self.const_grid_size_y = testing_vars.meta_coords.shape[3]

            zy_st, circular_flag = self.get_circular_flag(zy_st, zy_ed, self.const_grid_size_y)

            if not self.attention_flag:
                print("\n\t\t\tATTENTION: Pre sampling mode is deactivate\n")
                self.attention_flag = True
            partial = 0.8
            if hasattr(self.config.train_params, "partial"):
                partial = self.config.train_params.partial
            coords_partial = {
                "p_x_st": zx_st / self.const_grid_size_x,
                "p_x_ed": (zx_st + x_size) / self.const_grid_size_x,
                "p_y_st": zy_st / self.const_grid_size_y,
                "p_y_ed": (zy_st + y_size) / self.const_grid_size_y,
                "circular_flag": circular_flag,
                "x_total": self.const_grid_size_x,
                "y_total": self.const_grid_size_y,
                "test_flag": True,
                "start_flag": iiter == 0,
                "h_step": zx_st // 6,
                "w_step": zy_st // 6,
                "y_st": zy_st,
                "y_ed": zy_ed,
                # "full_shape": self.full_shape,
                "partial": partial,
            }
            g_ema_kwargs = {
                "global_latent": testing_vars.global_latent,
                "local_latent": cur_local_latent,
                "override_coords": cur_coords,
                "coords_partial_override": coords_partial,
                "noises": noises,
                "disable_dual_latents": True,
                "calc_flops": calc_flops,
            }
            # 得到上面这些参数就可以生成小块
            """
            coords_partial = {
                    "p_x_st": x_st / self.const_grid_size_x,
                    "p_x_ed": (x_st + x_size) / self.const_grid_size_x,
                    "p_y_st": y_st / self.const_grid_size_y,
                    "p_y_ed": (y_st + y_size) / self.const_grid_size_y,
                    "circular_flag": not coords[0, 1, 1, 0] == torch.min(coords[0, 1, :, :]),
                    "x_total": self.const_grid_size_x,
                    "y_total": self.const_grid_size_y 
                }
            """
            if hasattr(testing_vars, "wplus_styles") and testing_vars.wplus_styles is not None:
                g_ema_kwargs["wplus_styles"] = testing_vars.wplus_styles
            img_x_st, img_y_st = outfeat_x_st[-1], outfeat_y_st[-1]
            img_x_ed, img_y_ed = outfeat_x_ed[-1], outfeat_y_ed[-1]
            index_tuple = (img_x_st, img_x_ed, img_y_st, img_y_ed) # 0-101, 0-101
            # 下一句生成小块
            exec_time, flops = self.maybe_parallel_inference(
                testing_vars, g_ema_kwargs=g_ema_kwargs, index_tuple=index_tuple, return_exec_time=write_gpu_time, calc_flops=calc_flops)
            accum_exec_time += exec_time
            if calc_flops:
                accum_flops_all += flops["all"]
                accum_flops_ss += flops["ss"]
                accum_flops_ts += flops["ts"]

        exec_time, flops = self.maybe_parallel_inference(
            testing_vars, flush=True, return_exec_time=write_gpu_time, calc_flops=calc_flops)
        if calc_flops:
            accum_flops_all += flops["all"]
            accum_flops_ss += flops["ss"]
            accum_flops_ts += flops["ts"]

        if write_gpu_time:
            accum_exec_time += exec_time
            print(" [*] GPU time {}".format(accum_exec_time))
            self.accum_exec_times.append(accum_exec_time)
            fmt_date = datetime.date.today().strftime("%d-%m-%Y")
            benchmark_file = os.path.join(self.save_root, "speed_benchmark_{}.txt".format(fmt_date))
            with open(benchmark_file, "a") as f:
                f.write("{:.6f}".format(accum_exec_time))

        if calc_flops:
            print(" [*] Total FLOPs: {} (SS {}, TS {})".format(
                self.pretty_print_flops(accum_flops_all), 
                self.pretty_print_flops(accum_flops_ss), 
                self.pretty_print_flops(accum_flops_ts)))

    def circular_sample_width(self, tensor:torch.Tensor, y_width, x_st, x_ed, y_st, y_ed):
        """_summary_

        Args:
            tensor (torch.Tensor): value to be sampled
            y_width (int): total_width of the tensor to be sampled
            y_st (int): start point of the slice (non-circular)
            y_ed (int): end point of the slice (non-circular: may be beyond the x_width)
        """
        if y_ed <= y_width:
            # 不超出循环的情况
            return tensor[:, :, x_st:x_ed, y_st:y_ed]
        elif y_ed <= y_width * 2:
            # 超出循环一倍以内的情况
            if y_st < y_width:
                y_ed = y_ed % y_width
                return torch.cat((tensor[:, :, x_st:x_ed, y_st:], tensor[:, :, x_st:x_ed, :y_ed]), dim=3)
            else:
                y_st = y_st % y_width
                y_ed = y_ed % y_width
                return tensor[:, :, x_st:x_ed, y_st:y_ed]
        else:
            # 超出循环两倍及以上进行递归求解
            assert y_st > y_width, "width should be larger than 35"
            y_st = y_st - y_width
            y_ed = y_ed - y_width
            return self.circular_sample_width(tensor, y_width, x_st, x_ed, y_st, y_ed)

    def save_results(self, meta_img, dump_vars=None):
        print(" [*] Saving results...")
        self.save_meta_imgs(meta_img)
        if dump_vars is not None:
            self.save_testing_vars(dump_vars)
        self.cur_global_id += self.config.train_params.batch_size

    def save_testing_vars(self, testing_vars):
        assert self.config.train_params.batch_size == 1, \
            "This is only designed to be used with the interactive tool."
        save_path = os.path.join(self.save_root, str(self.cur_global_id).zfill(6)+".pkl")
        pkl.dump(testing_vars, open(save_path, "wb"))

    def _wrap_feature(self, feat, wrap_size, dim):
        assert wrap_size < (feat.shape[dim] - 2*wrap_size), \
            "Does not expect the wrapping area is larger than a full period."
        if dim == 2:
            valid_st = feat[:, :, wrap_size:2*wrap_size]
            valid_ed = feat[:, :, -2*wrap_size:-wrap_size]
            feat[:, :, :wrap_size] = valid_ed
            feat[:, :, -wrap_size:] = valid_st
        elif dim == 3:
            valid_st = feat[:, :, :, wrap_size:2*wrap_size]
            valid_ed = feat[:, :, :, -2*wrap_size:-wrap_size]
            feat[:, :, :, :wrap_size] = valid_ed
            feat[:, :, :, -wrap_size:] = valid_st
        else:
            raise NotImplementedError(
                "I don't expect this function will be used other than spatial dims, but got {}.".format(dim))
        
    def save_meta_imgs(self, meta_img):
        self.full_image = meta_img
        # Center crop
        pad_h = (self.meta_height - self.target_height) // 2
        pad_w = (self.meta_width - self.target_width) // 2
        meta_img = meta_img[:, :, pad_h:pad_h+self.target_height, pad_w:pad_w+self.target_width]

        # Save the full image and the low-resolution image (for visualization)
        meta_img = meta_img.clamp(-1, 1).permute(0, 2, 3, 1)
        meta_img = (meta_img + 1) / 2
        meta_img_np = meta_img.numpy()
        
        for i in range(self.config.train_params.batch_size):
            global_id = self.cur_global_id + i
            save_path = os.path.join(self.save_root, str(global_id).zfill(6)+".png")
            plt.imsave(save_path, meta_img_np[i])

    def _create_start_pts_mesh(self, step_size, num_steps_h, num_steps_w):
        # 定义重叠的尺寸
        start_pts_x = np.arange(num_steps_h) * step_size
        start_pts_y = np.arange(num_steps_w) * step_size
        start_pts_mesh = np.stack([
            np.repeat(start_pts_x.reshape(num_steps_h, 1), num_steps_w, axis=1),
            np.repeat(start_pts_y.reshape(1, num_steps_w), num_steps_h, axis=0),
        ], 2).astype(np.uint32) # shape: (H, W, 2)
        return start_pts_mesh

    def _init_starting_points(self):
        # 定义重叠的尺寸
        # Get all the starting points of the patches to generate, 计算生成的小块数量
        self.num_steps_h = \
            math.ceil((self.target_height - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad
        self.num_steps_w = \
            math.ceil((self.target_width  - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad

        self.start_pts_mesh_z = self._create_start_pts_mesh(
            step_size=self.latentspace_step_size, 
            num_steps_h=self.num_steps_h,
            num_steps_w=self.num_steps_w)
        self.start_pts_mesh_z += self.ss_unfold_size

        # Create this for:
        # (1) Final image pixels assignment
        # (2) Randomized noise handling within the texture synthesizer
        self.start_pts_mesh_outfeats = [
            self._create_start_pts_mesh(
                step_size=step_size,
                num_steps_h=self.num_steps_h,
                num_steps_w=self.num_steps_w,
            ) for step_size in self.outfeat_step_sizes]
        # start_pts_mesh_x = \
        #     (start_pts_mesh_z - ss_unfold_size) // latentspace_step_size * pixelspace_step_size # shape: (H, W, 2)
        # start_pts_mesh_x = start_pts_mesh_x.astype(np.uint32)

        # To avoid edge-condition on the image edge, we generate an image slightly larger than
        # requested, then center-crop to the requested resolution.
        self.meta_height = self.pixelspace_step_size * (self.num_steps_h-1) + self.outfeat_sizes_list[-1]
        self.meta_width  = self.pixelspace_step_size * (self.num_steps_w-1) + self.outfeat_sizes_list[-1]
        # height=485, width=773 (256 x 512)
        # height=485, width=965 (256 x 768)

    def compose_inv_root(self):
        return os.path.join("./logs/", self.config.var.exp_name, "test", self.config.task.prev_inv_config, "stats")

    def _init_close_loop_starting_points(self):
        # 定义重叠的尺寸
        # Get all the starting points of the patches to generate, 计算生成的小块数量
        self.num_steps_h = \
            math.ceil((self.target_height - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad
        # self.num_steps_w = \
        #     math.ceil((self.target_width  - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad
        assert self.target_width % self.pixelspace_step_size == 0, f"close-loop property is not ensured\n target width {self.target_width} is not divided by target patch size {self.pixelspace_step_size}"
        self.num_steps_w_min = math.ceil(self.target_width / self.pixelspace_step_size)
        if self.target_width == 1152:
            self.num_steps_w = self.num_steps_w_min + 2
            # for test
        else:
            self.num_steps_w = self.num_steps_w_min + 2
        # 多出的步数为重叠回来的步数
        
        self.start_pts_mesh_z = self._create_start_pts_mesh(
            step_size=self.latentspace_step_size, 
            num_steps_h=self.num_steps_h,
            num_steps_w=self.num_steps_w)
        self.start_pts_mesh_z += self.ss_unfold_size

        # Create this for:
        # (1) Final image pixels assignment
        # (2) Randomized noise handling within the texture synthesizer
        self.start_pts_mesh_outfeats = [
            self._create_start_pts_mesh(
                step_size=step_size,
                num_steps_h=self.num_steps_h,
                num_steps_w=self.num_steps_w,
            ) for step_size in self.outfeat_step_sizes]
        # start_pts_mesh_x = \
        #     (start_pts_mesh_z - ss_unfold_size) // latentspace_step_size * pixelspace_step_size # shape: (H, W, 2)
        # start_pts_mesh_x = start_pts_mesh_x.astype(np.uint32)

        # To avoid edge-condition on the image edge, we generate an image slightly larger than
        # requested, then center-crop to the requested resolution.
        self.meta_height = self.pixelspace_step_size * (self.num_steps_h-1) + self.outfeat_sizes_list[-1]
        self.meta_width  = self.num_steps_w_min * self.pixelspace_step_size
        # height=485, width=773 (256 x 512)
        # height=485, width=965 (256 x 768)

    def get_circular_flag(self, zy_st, zy_ed, y_total):
        # 训练时可以保证zy_st不超出坐标的取值范围, 测试时无法保证
        if zy_ed > y_total:
            if zy_st < y_total:
                circular_flag = True
                return zy_st, circular_flag
            else:
                circular_flag = False
                return zy_st % y_total, circular_flag
        else:
            circular_flag = False
            return zy_st, circular_flag


# 隐变量插值实验挑图
class InfiniteGenerationManagerPatchCoordsCloseLoopInterpolateLoad(BaseTestManager):
    attention_flag = False
    def save_full_imgs(self):
        # Save the full image and the low-resolution image (for visualization)
        meta_img = self.full_image
        meta_img = meta_img.clamp(-1, 1).permute(0, 2, 3, 1)
        meta_img = (meta_img + 1) / 2
        meta_img_np = meta_img.numpy()
        
        for i in range(self.config.train_params.batch_size):
            global_id = self.cur_global_id + i - self.config.train_params.batch_size
            # save after adding the global_id
            save_path = os.path.join(self.save_root, str(global_id).zfill(6)+"full.png")
            plt.imsave(save_path, meta_img_np[i])
            
    def task_specific_init(self, output_size=None, **kwargs):

        if output_size is None:
            self.target_height = self.config.task.height
            self.target_width = self.config.task.width
        else:
            self.target_height, self.target_width = output_size
           
        self._init_close_loop_starting_points()         
        self.noise_heights = self.outfeat_step_sizes * (self.num_steps_h-1) + self.outfeat_sizes_list
        self.noise_widths  = self.outfeat_step_sizes * self.num_steps_w_min

        if hasattr(self.config.task, "gen_from_inv_stats") and self.config.task.gen_from_inv_stats:
            self.inv_root = self.compose_inv_root()
            self.inv_rec_files = sorted(glob(os.path.join(self.inv_root, "*")))
            self.gen_from_inv_stats = True
        else:
            self.gen_from_inv_stats = False

        self._sample_inter(seed=kwargs.get("seed", None))

    def _load_latent_ckpt(self, path, _id):
        ckpt = torch.load(path)
        # self.save_dict["local_latent"] = local_latent
        # self.save_dict["global_latent"] = global_latent
        # self.save_dict["noises"] = randomized_noises
        batch_random_noises = ckpt["noises"]
        random_noises = [e[_id, :, :, :].unsqueeze(0) for e in batch_random_noises]
        return (
            ckpt['global_latent'][_id, :].unsqueeze(0), 
            ckpt["local_latent"][_id, :, :, :].unsqueeze(0), 
            random_noises,
        )

    def _sample_inter(self, seed):
        assert seed is not None
        if seed == 1:
            load_path1 = "/mnt/disk10T/sxc/360_gen/infinityGAN/test_interpolate/312-319.pth.tar"
            _id1 = 6
            load_path2 = "/mnt/disk10T/sxc/360_gen/infinityGAN/test_interpolate/392-399.pth.tar"
            _id2 = 2
        elif seed == 2:
            # 2130 - 2480 有问题
            load_path1 = "/mnt/disk10T/sxc/360_gen/infinityGAN/test_interpolate/2128-2135.pth.tar"
            _id1 = 2
            load_path2 = "/mnt/disk10T/sxc/360_gen/infinityGAN/test_interpolate/2480-2487.pth.tar"
            _id2 = 0
        elif seed == 3:
            load_path1 = "/mnt/disk10T/sxc/360_gen/infinityGAN/test_interpolate/2696-2703.pth.tar"
            _id1 = 5
            load_path2 = "/mnt/disk10T/sxc/360_gen/infinityGAN/test_interpolate/8744-8751.pth.tar"
            _id2 = 6
        elif seed == 4:
            load_path1 = "/mnt/disk10T/sxc/360_gen/infinityGAN/test_interpolate/8920-8927.pth.tar"
            _id1 = 2
            load_path2 = "/mnt/disk10T/sxc/360_gen/infinityGAN/test_interpolate/9120-9127.pth.tar"
            _id2 = 6
        elif seed == 5:
            load_path1 = "/mnt/disk10T/sxc/360_gen/infinityGAN/test_interpolate/8744-8751.pth.tar"
            _id1 = 6
            load_path2 = "/mnt/disk10T/sxc/360_gen/infinityGAN/test_interpolate/4160-4167.pth.tar"
            _id2 = 6
        elif seed == 6:
            # 2806 - 2480
            load_path1 = "/mnt/disk10T/sxc/360_gen/infinityGAN/test_interpolate/2800-2807.pth.tar"
            _id1 = 6
            load_path2 = "/mnt/disk10T/sxc/360_gen/infinityGAN/test_interpolate/2480-2487.pth.tar"
            _id2 = 0
        else:
            raise ValueError(f"exp error, seed is {seed}")
        
        self.global_1, self.local_1, self.r_noise_1 = self._load_latent_ckpt(load_path1, _id1)
        self.global_2, self.local_2, self.r_noise_2 = self._load_latent_ckpt(load_path2, _id2)

    
    def run_next(self, save=True, write_gpu_time=False, inv_records=None, inv_placements=None, calc_flops=False, disable_pbar=False, **kwargs):
        if len(kwargs) > 0:
            for k,v in kwargs.items():
                if v is not None and k not in ["seeds"]:
                    print(" [Warning] task manager receives untracked arg {} with value {}".format(k ,v))
        testing_vars = self.create_vars(inv_records=inv_records, inv_placements=inv_placements, seed=kwargs.get("seeds", None), _iter=kwargs.get("_iter", None))
        self.generate(testing_vars, write_gpu_time=write_gpu_time, calc_flops=calc_flops, disable_pbar=disable_pbar)
        if save:
            self.save_results(testing_vars.meta_img)
        return testing_vars.meta_img

    def create_vars(self, inv_records=None, inv_placements=None, seed=None, _iter=None):
        # 对local_latent进行采样
        assert _iter is not None, "Please specify the iteration number for the interpolation."
        mixing = False
        assert mixing == False, "Otherwise, an injection index must be specified and fed into g_ema."
        
        # Allocate memory for the final output, starts iterating and filling in the generated results.
        # Can be reused
        meta_img = torch.empty(
            self.config.train_params.batch_size,
            3,
            int(self.meta_height),
            int(self.meta_width)).float()

        # [Note]
        # 1.  One may implement a sophisticated version that does not required to 
        #     generate all the latents at once, as most of the info are not reusing
        #     during the inference. However, the author is just lazy and abusing his 
        #     CPU memory OuO
        # global_latent = self.latent_sampler.sample_global_latent(
        #     self.config.train_params.batch_size, mixing=mixing, device=self.device, seed=seed)
        # full_local_latent_shape = (
        #     # Does not account GNN padding here, it is handled within the latent_sampler
        #     int(self.g_ema_module.calc_in_spatial_size(self.meta_height, include_ss=False)),
        #     int(self.g_ema_module.calc_in_spatial_size(self.meta_width, include_ss=False)),
        # )
        # local_latent = self.latent_sampler.sample_local_latent(
        #     self.config.train_params.batch_size, 
        #     device="cpu", # Store in CPU anyway, it can be INFINITLY LARGE!
        #     specific_shape=full_local_latent_shape)

        # height = self.g_ema_module.calc_in_spatial_size(self.meta_height, include_ss=False)

        # local_latent = self.latent_sampler.sample_circular_local_latent_patch101(
        #     self.config.train_params.batch_size,
        #     device="cpu",
        #     meta_width=self.meta_width,
        #     height_in=height,
        #     seed=seed,
        # )
        alpha = np.linspace(0, 1, self.config.task.num_gen)[_iter]
        global_latent = (1 - alpha) * self.global_1 + alpha * self.global_2
        local_latent = (1 - alpha) * self.local_1 + alpha * self.local_2

        self.full_shape = local_latent.shape[2:]

        meta_coords = self.coord_handler.sample_coord_grid(
            local_latent, 
            is_training=False) # 通过local_latent得到坐标

        # Create randomized noises, 得到随机噪声
        # 噪声也由seed控制，保证一样
        # if seed is not None:
        #     randomized_noises = [
        #         torch.from_numpy(np.random.RandomState(seed).randn(self.config.train_params.batch_size, 1, int(h), int(w)))
        #             for (h,w) in zip(self.noise_heights, self.noise_widths)]
        # else:
        randomized_noises = [(1 - alpha) * e1 + alpha * e2 for (e1,e2) in zip(self.r_noise_1, self.r_noise_2)]

        testing_vars = TestingVars(
            meta_img=meta_img, 
            global_latent=global_latent, 
            local_latent=local_latent, 
            meta_coords=meta_coords, 
            noises=randomized_noises, 
            device=self.device)
        # 通过testing vars对测试的参数进行封装

        if self.gen_from_inv_stats:
            assert inv_records is None, \
                "`gen_from_inv_stats` already specified, should not receive `inv_records` from command!"
            assert self.config.train_params.batch_size == 1, \
                "Inverted parameters loading for batch is not yet implemeted! " + \
                "Please use parallel-batching instead, which provides a similar inference speed."
            inv_records = [self.inv_rec_files[self.cur_global_id]]
            inv_placements = [self.config.task.gen_from_inv_placement]

        if inv_records is not None:
            testing_vars.replace_by_records(
                self.g_ema_module, inv_records, inv_placements, assert_no_style=True)
        
        return testing_vars

    def generate(self, testing_vars, tkinter_pbar=None, update_by_ss_map=None, update_by_ts_map=None, 
                 write_gpu_time=False, calc_flops=False, disable_pbar=False):

        # I don't mind bruteforce casting combination here, cuz you should worry about the meta_img size first
        idx_tuples = list(iter_product(range(self.start_pts_mesh_z.shape[0]), range(self.start_pts_mesh_z.shape[1])))

        if disable_pbar:
            pbar = idx_tuples
        elif tkinter_pbar is not None:
            pbar = tkinter_pbar(idx_tuples)
        else:
            pbar = tqdm(idx_tuples)

        accum_exec_time = 0
        accum_flops_all, accum_flops_ss, accum_flops_ts = 0, 0, 0
        for iiter, (idx_x,idx_y) in enumerate(pbar):
            zx_st, zy_st = self.start_pts_mesh_z[idx_x, idx_y]
            zx_ed = zx_st + self.config.train_params.ts_input_size 
            zy_ed = zy_st + self.config.train_params.ts_input_size

            # Handle the randomized noise input of the texture_synthesizer...
            outfeat_x_st = [start_pts_mesh[idx_x,idx_y,0] for start_pts_mesh in self.start_pts_mesh_outfeats]
            outfeat_y_st = [start_pts_mesh[idx_x,idx_y,1] for start_pts_mesh in self.start_pts_mesh_outfeats]
            outfeat_x_ed = [
                x_st + out_size for (x_st, out_size) in zip(outfeat_x_st, self.outfeat_sizes_list)]
            outfeat_y_ed = [
                y_st + out_size for (y_st, out_size) in zip(outfeat_y_st, self.outfeat_sizes_list)]
            noises = []
            for i, (fx_st, fy_st, fx_ed, fy_ed) in enumerate(zip(outfeat_x_st, outfeat_y_st, outfeat_x_ed, outfeat_y_ed)):
                # noises.append(testing_vars.noises[i][:, :, fx_st:fx_ed, fy_st:fy_ed].to(self.device))
                noises.append(self.circular_sample_width(
                    testing_vars.noises[i],
                    self.noise_widths[i],
                    fx_st,
                    fx_ed, 
                    fy_st,
                    fy_ed,
                    ).to(self.device))
            # 0:19, 0:17, 0:31, 0:29, 0:55, 0:53, 0:103, 0:101; x,y相同, 由start_pts_mesh控制
            # x相同, y为 12:31, 12:29, 24:55, 24:53, 48:103, 48:101, 96:199, 96:197  
            # noise 到 meta_height x meta_width, 和local_latent相对应, 这里偏置为12
            # local_latent的偏置为6
            # Deal with SS unfolding here
            zx_st -= self.ss_unfold_size
            zy_st -= self.ss_unfold_size
            zx_ed += self.ss_unfold_size
            zy_ed += self.ss_unfold_size
            
            # [Interactive] Decide whether the region will be updated, otherwise no need to generate
            if update_by_ss_map is not None:
                ss_cursors = zx_st, zx_ed, zy_st, zy_ed
                if not self.is_overlaping_update_map(update_by_ss_map, *ss_cursors):
                    continue
            if update_by_ts_map is not None:
                # For TS regional selection, we only select noises
                ts_cursors = outfeat_x_st[0], outfeat_x_ed[0], outfeat_y_st[0], outfeat_y_ed[0]
                if not self.is_overlaping_update_map(update_by_ts_map, *ts_cursors):
                    continue

            # cur_local_latent = testing_vars.local_latent[:, :, zx_st:zx_ed, zy_st:zy_ed].to(self.device)
            # # 0:35, 0:35; 下一循环为 0:35, 6:41
            # cur_coords = testing_vars.meta_coords[:, :, zx_st:zx_ed, zy_st:zy_ed].to(self.device)
            cur_local_latent = self.circular_sample_width(testing_vars.local_latent, testing_vars.local_latent.shape[3], zx_st, zx_ed, zy_st, zy_ed).to(self.device)
            cur_coords = self.circular_sample_width(testing_vars.meta_coords, testing_vars.meta_coords.shape[3], zx_st, zx_ed, zy_st, zy_ed).to(self.device)

            x_size = zx_ed - zx_st + 1
            y_size = zy_ed - zy_st + 1
            
            
            self.const_grid_size_x = testing_vars.meta_coords.shape[2]
            self.const_grid_size_y = testing_vars.meta_coords.shape[3]

            zy_st, circular_flag = self.get_circular_flag(zy_st, zy_ed, self.const_grid_size_y)

            if not self.attention_flag:
                print("\n\t\t\tATTENTION: Pre sampling mode is deactivate\n")
                self.attention_flag = True
            partial = 0.8
            if hasattr(self.config.train_params, "partial"):
                partial = self.config.train_params.partial
            coords_partial = {
                "p_x_st": zx_st / self.const_grid_size_x,
                "p_x_ed": (zx_st + x_size) / self.const_grid_size_x,
                "p_y_st": zy_st / self.const_grid_size_y,
                "p_y_ed": (zy_st + y_size) / self.const_grid_size_y,
                "circular_flag": circular_flag,
                "x_total": self.const_grid_size_x,
                "y_total": self.const_grid_size_y,
                "test_flag": True,
                "start_flag": iiter == 0,
                "h_step": zx_st // 6,
                "w_step": zy_st // 6,
                "y_st": zy_st,
                "y_ed": zy_ed,
                # "full_shape": self.full_shape,
                "partial": partial,
            }
            g_ema_kwargs = {
                "global_latent": testing_vars.global_latent,
                "local_latent": cur_local_latent,
                "override_coords": cur_coords,
                "coords_partial_override": coords_partial,
                "noises": noises,
                "disable_dual_latents": True,
                "calc_flops": calc_flops,
            }
            # 得到上面这些参数就可以生成小块
            """
            coords_partial = {
                    "p_x_st": x_st / self.const_grid_size_x,
                    "p_x_ed": (x_st + x_size) / self.const_grid_size_x,
                    "p_y_st": y_st / self.const_grid_size_y,
                    "p_y_ed": (y_st + y_size) / self.const_grid_size_y,
                    "circular_flag": not coords[0, 1, 1, 0] == torch.min(coords[0, 1, :, :]),
                    "x_total": self.const_grid_size_x,
                    "y_total": self.const_grid_size_y 
                }
            """
            if hasattr(testing_vars, "wplus_styles") and testing_vars.wplus_styles is not None:
                g_ema_kwargs["wplus_styles"] = testing_vars.wplus_styles
            img_x_st, img_y_st = outfeat_x_st[-1], outfeat_y_st[-1]
            img_x_ed, img_y_ed = outfeat_x_ed[-1], outfeat_y_ed[-1]
            index_tuple = (img_x_st, img_x_ed, img_y_st, img_y_ed) # 0-101, 0-101
            # 下一句生成小块
            exec_time, flops = self.maybe_parallel_inference(
                testing_vars, g_ema_kwargs=g_ema_kwargs, index_tuple=index_tuple, return_exec_time=write_gpu_time, calc_flops=calc_flops)
            accum_exec_time += exec_time
            if calc_flops:
                accum_flops_all += flops["all"]
                accum_flops_ss += flops["ss"]
                accum_flops_ts += flops["ts"]

        exec_time, flops = self.maybe_parallel_inference(
            testing_vars, flush=True, return_exec_time=write_gpu_time, calc_flops=calc_flops)
        if calc_flops:
            accum_flops_all += flops["all"]
            accum_flops_ss += flops["ss"]
            accum_flops_ts += flops["ts"]

        if write_gpu_time:
            accum_exec_time += exec_time
            print(" [*] GPU time {}".format(accum_exec_time))
            self.accum_exec_times.append(accum_exec_time)
            fmt_date = datetime.date.today().strftime("%d-%m-%Y")
            benchmark_file = os.path.join(self.save_root, "speed_benchmark_{}.txt".format(fmt_date))
            with open(benchmark_file, "a") as f:
                f.write("{:.6f}".format(accum_exec_time))

        if calc_flops:
            print(" [*] Total FLOPs: {} (SS {}, TS {})".format(
                self.pretty_print_flops(accum_flops_all), 
                self.pretty_print_flops(accum_flops_ss), 
                self.pretty_print_flops(accum_flops_ts)))

    def circular_sample_width(self, tensor:torch.Tensor, y_width, x_st, x_ed, y_st, y_ed):
        """_summary_

        Args:
            tensor (torch.Tensor): value to be sampled
            y_width (int): total_width of the tensor to be sampled
            y_st (int): start point of the slice (non-circular)
            y_ed (int): end point of the slice (non-circular: may be beyond the x_width)
        """
        if y_ed <= y_width:
            # 不超出循环的情况
            return tensor[:, :, x_st:x_ed, y_st:y_ed]
        elif y_ed <= y_width * 2:
            # 超出循环一倍以内的情况
            if y_st < y_width:
                y_ed = y_ed % y_width
                return torch.cat((tensor[:, :, x_st:x_ed, y_st:], tensor[:, :, x_st:x_ed, :y_ed]), dim=3)
            else:
                y_st = y_st % y_width
                y_ed = y_ed % y_width
                return tensor[:, :, x_st:x_ed, y_st:y_ed]
        else:
            # 超出循环两倍及以上进行递归求解
            assert y_st > y_width, "width should be larger than 35"
            y_st = y_st - y_width
            y_ed = y_ed - y_width
            return self.circular_sample_width(tensor, y_width, x_st, x_ed, y_st, y_ed)

    def save_results(self, meta_img, dump_vars=None):
        print(" [*] Saving results...")
        self.save_meta_imgs(meta_img)
        if dump_vars is not None:
            self.save_testing_vars(dump_vars)
        self.cur_global_id += self.config.train_params.batch_size

    def save_testing_vars(self, testing_vars):
        assert self.config.train_params.batch_size == 1, \
            "This is only designed to be used with the interactive tool."
        save_path = os.path.join(self.save_root, str(self.cur_global_id).zfill(6)+".pkl")
        pkl.dump(testing_vars, open(save_path, "wb"))

    def _wrap_feature(self, feat, wrap_size, dim):
        assert wrap_size < (feat.shape[dim] - 2*wrap_size), \
            "Does not expect the wrapping area is larger than a full period."
        if dim == 2:
            valid_st = feat[:, :, wrap_size:2*wrap_size]
            valid_ed = feat[:, :, -2*wrap_size:-wrap_size]
            feat[:, :, :wrap_size] = valid_ed
            feat[:, :, -wrap_size:] = valid_st
        elif dim == 3:
            valid_st = feat[:, :, :, wrap_size:2*wrap_size]
            valid_ed = feat[:, :, :, -2*wrap_size:-wrap_size]
            feat[:, :, :, :wrap_size] = valid_ed
            feat[:, :, :, -wrap_size:] = valid_st
        else:
            raise NotImplementedError(
                "I don't expect this function will be used other than spatial dims, but got {}.".format(dim))
        
    def save_meta_imgs(self, meta_img):
        self.full_image = meta_img
        # Center crop
        pad_h = (self.meta_height - self.target_height) // 2
        pad_w = (self.meta_width - self.target_width) // 2
        meta_img = meta_img[:, :, pad_h:pad_h+self.target_height, pad_w:pad_w+self.target_width]

        # Save the full image and the low-resolution image (for visualization)
        meta_img = meta_img.clamp(-1, 1).permute(0, 2, 3, 1)
        meta_img = (meta_img + 1) / 2
        meta_img_np = meta_img.numpy()
        
        for i in range(self.config.train_params.batch_size):
            global_id = self.cur_global_id + i
            save_path = os.path.join(self.save_root, str(global_id).zfill(6)+".png")
            plt.imsave(save_path, meta_img_np[i])

    def _create_start_pts_mesh(self, step_size, num_steps_h, num_steps_w):
        # 定义重叠的尺寸
        start_pts_x = np.arange(num_steps_h) * step_size
        start_pts_y = np.arange(num_steps_w) * step_size
        start_pts_mesh = np.stack([
            np.repeat(start_pts_x.reshape(num_steps_h, 1), num_steps_w, axis=1),
            np.repeat(start_pts_y.reshape(1, num_steps_w), num_steps_h, axis=0),
        ], 2).astype(np.uint32) # shape: (H, W, 2)
        return start_pts_mesh

    def _init_starting_points(self):
        # 定义重叠的尺寸
        # Get all the starting points of the patches to generate, 计算生成的小块数量
        self.num_steps_h = \
            math.ceil((self.target_height - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad
        self.num_steps_w = \
            math.ceil((self.target_width  - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad

        self.start_pts_mesh_z = self._create_start_pts_mesh(
            step_size=self.latentspace_step_size, 
            num_steps_h=self.num_steps_h,
            num_steps_w=self.num_steps_w)
        self.start_pts_mesh_z += self.ss_unfold_size

        # Create this for:
        # (1) Final image pixels assignment
        # (2) Randomized noise handling within the texture synthesizer
        self.start_pts_mesh_outfeats = [
            self._create_start_pts_mesh(
                step_size=step_size,
                num_steps_h=self.num_steps_h,
                num_steps_w=self.num_steps_w,
            ) for step_size in self.outfeat_step_sizes]
        # start_pts_mesh_x = \
        #     (start_pts_mesh_z - ss_unfold_size) // latentspace_step_size * pixelspace_step_size # shape: (H, W, 2)
        # start_pts_mesh_x = start_pts_mesh_x.astype(np.uint32)

        # To avoid edge-condition on the image edge, we generate an image slightly larger than
        # requested, then center-crop to the requested resolution.
        self.meta_height = self.pixelspace_step_size * (self.num_steps_h-1) + self.outfeat_sizes_list[-1]
        self.meta_width  = self.pixelspace_step_size * (self.num_steps_w-1) + self.outfeat_sizes_list[-1]
        # height=485, width=773 (256 x 512)
        # height=485, width=965 (256 x 768)

    def compose_inv_root(self):
        return os.path.join("./logs/", self.config.var.exp_name, "test", self.config.task.prev_inv_config, "stats")

    def _init_close_loop_starting_points(self):
        # 定义重叠的尺寸
        # Get all the starting points of the patches to generate, 计算生成的小块数量
        self.num_steps_h = \
            math.ceil((self.target_height - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad
        # self.num_steps_w = \
        #     math.ceil((self.target_width  - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad
        assert self.target_width % self.pixelspace_step_size == 0, f"close-loop property is not ensured\n target width {self.target_width} is not divided by target patch size {self.pixelspace_step_size}"
        self.num_steps_w_min = math.ceil(self.target_width / self.pixelspace_step_size)
        if self.target_width == 1152:
            self.num_steps_w = self.num_steps_w_min + 2
            # for test
        else:
            self.num_steps_w = self.num_steps_w_min + 2
        # 多出的步数为重叠回来的步数
        
        self.start_pts_mesh_z = self._create_start_pts_mesh(
            step_size=self.latentspace_step_size, 
            num_steps_h=self.num_steps_h,
            num_steps_w=self.num_steps_w)
        self.start_pts_mesh_z += self.ss_unfold_size

        # Create this for:
        # (1) Final image pixels assignment
        # (2) Randomized noise handling within the texture synthesizer
        self.start_pts_mesh_outfeats = [
            self._create_start_pts_mesh(
                step_size=step_size,
                num_steps_h=self.num_steps_h,
                num_steps_w=self.num_steps_w,
            ) for step_size in self.outfeat_step_sizes]
        # start_pts_mesh_x = \
        #     (start_pts_mesh_z - ss_unfold_size) // latentspace_step_size * pixelspace_step_size # shape: (H, W, 2)
        # start_pts_mesh_x = start_pts_mesh_x.astype(np.uint32)

        # To avoid edge-condition on the image edge, we generate an image slightly larger than
        # requested, then center-crop to the requested resolution.
        self.meta_height = self.pixelspace_step_size * (self.num_steps_h-1) + self.outfeat_sizes_list[-1]
        self.meta_width  = self.num_steps_w_min * self.pixelspace_step_size
        # height=485, width=773 (256 x 512)
        # height=485, width=965 (256 x 768)

    def get_circular_flag(self, zy_st, zy_ed, y_total):
        # 训练时可以保证zy_st不超出坐标的取值范围, 测试时无法保证
        if zy_ed > y_total:
            if zy_st < y_total:
                circular_flag = True
                return zy_st, circular_flag
            else:
                circular_flag = False
                return zy_st % y_total, circular_flag
        else:
            circular_flag = False
            return zy_st, circular_flag



# patch coords 用于球卷积的生成器测试
class InfiniteGenerationManagerPatchCoordsCloseLoopFast(BaseTestManager):
    attention_flag = False
    def save_full_imgs(self):
        # Save the full image and the low-resolution image (for visualization)
        meta_img = self.full_image
        meta_img = meta_img.clamp(-1, 1).permute(0, 2, 3, 1)
        meta_img = (meta_img + 1) / 2
        meta_img_np = meta_img.numpy()
        
        for i in range(self.config.train_params.batch_size):
            global_id = self.cur_global_id + i - self.config.train_params.batch_size
            # save after adding the global_id
            save_path = os.path.join(self.save_root, str(global_id).zfill(6)+"full.png")
            plt.imsave(save_path, meta_img_np[i])
            
    def task_specific_init(self, output_size=None, **kwargs):

        if output_size is None:
            self.target_height = self.config.task.height
            self.target_width = self.config.task.width
        else:
            self.target_height, self.target_width = output_size
           
        self._init_close_loop_starting_points()         
        self.noise_heights = self.outfeat_step_sizes * (self.num_steps_h-1) + self.outfeat_sizes_list
        self.noise_widths  = self.outfeat_step_sizes * self.num_steps_w_min

        if hasattr(self.config.task, "gen_from_inv_stats") and self.config.task.gen_from_inv_stats:
            self.inv_root = self.compose_inv_root()
            self.inv_rec_files = sorted(glob(os.path.join(self.inv_root, "*")))
            self.gen_from_inv_stats = True
        else:
            self.gen_from_inv_stats = False

        self.save = False
        # print("save all random noises")
        # self.save_dict = {}
        # self.latent_save_path = os.path.join(self.save_root, "latents")
        # os.makedirs(self.latent_save_path, exist_ok=True)

    def run_next(self, save=True, write_gpu_time=False, inv_records=None, inv_placements=None, calc_flops=False, disable_pbar=False, **kwargs):
        if len(kwargs) > 0:
            for k,v in kwargs.items():
                if v is not None and k not in ["seeds"]:
                    print(" [Warning] task manager receives untracked arg {} with value {}".format(k ,v))
        testing_vars = self.create_vars(inv_records=inv_records, inv_placements=inv_placements, seed=kwargs.get("seeds", None))
        self.generate(testing_vars, write_gpu_time=write_gpu_time, calc_flops=calc_flops, disable_pbar=disable_pbar)
        if save:
            self.save_results(testing_vars.meta_img)
        return testing_vars.meta_img

    def create_vars(self, inv_records=None, inv_placements=None, seed=None):
        # 对local_latent进行采样
        mixing = False
        assert mixing == False, "Otherwise, an injection index must be specified and fed into g_ema."
        
        # Allocate memory for the final output, starts iterating and filling in the generated results.
        # Can be reused
        meta_img = torch.empty(
            self.config.train_params.batch_size,
            3,
            int(self.meta_height),
            int(self.meta_width)).float()

        # [Note]
        # 1.  One may implement a sophisticated version that does not required to 
        #     generate all the latents at once, as most of the info are not reusing
        #     during the inference. However, the author is just lazy and abusing his 
        #     CPU memory OuO
        global_latent = self.latent_sampler.sample_global_latent(
            self.config.train_params.batch_size, mixing=mixing, device=self.device, seed=seed)
        # full_local_latent_shape = (
        #     # Does not account GNN padding here, it is handled within the latent_sampler
        #     int(self.g_ema_module.calc_in_spatial_size(self.meta_height, include_ss=False)),
        #     int(self.g_ema_module.calc_in_spatial_size(self.meta_width, include_ss=False)),
        # )
        # local_latent = self.latent_sampler.sample_local_latent(
        #     self.config.train_params.batch_size, 
        #     device="cpu", # Store in CPU anyway, it can be INFINITLY LARGE!
        #     specific_shape=full_local_latent_shape)

        height = self.g_ema_module.calc_in_spatial_size(self.meta_height, include_ss=False)

        local_latent = self.latent_sampler.sample_circular_local_latent_patch101(
            self.config.train_params.batch_size,
            device="cpu",
            meta_width=self.meta_width,
            height_in=height,
            seed=seed,
        )

        self.full_shape = local_latent.shape[2:]

        meta_coords = self.coord_handler.sample_coord_grid(
            local_latent, 
            is_training=False) # 通过local_latent得到坐标

        # Create randomized noises, 得到随机噪声
        # 噪声也由seed控制，保证一样
        # if seed is not None:
        #     randomized_noises = [
        #         torch.from_numpy(np.random.RandomState(seed).randn(self.config.train_params.batch_size, 1, int(h), int(w)))
        #             for (h,w) in zip(self.noise_heights, self.noise_widths)]
        # else:
        randomized_noises = [
            torch.randn(self.config.train_params.batch_size, 1, int(h), int(w))
                for (h,w) in zip(self.noise_heights, self.noise_widths)]

        if self.save:
            self.save_dict["local_latent"] = local_latent
            self.save_dict["global_latent"] = global_latent
            self.save_dict["noises"] = randomized_noises

        testing_vars = TestingVars(
            meta_img=meta_img, 
            global_latent=global_latent, 
            local_latent=local_latent, 
            meta_coords=meta_coords, 
            noises=randomized_noises, 
            device=self.device)
        # 通过testing vars对测试的参数进行封装

        if self.gen_from_inv_stats:
            assert inv_records is None, \
                "`gen_from_inv_stats` already specified, should not receive `inv_records` from command!"
            assert self.config.train_params.batch_size == 1, \
                "Inverted parameters loading for batch is not yet implemeted! " + \
                "Please use parallel-batching instead, which provides a similar inference speed."
            inv_records = [self.inv_rec_files[self.cur_global_id]]
            inv_placements = [self.config.task.gen_from_inv_placement]

        if inv_records is not None:
            testing_vars.replace_by_records(
                self.g_ema_module, inv_records, inv_placements, assert_no_style=True)
        
        return testing_vars

    def generate(self, testing_vars, tkinter_pbar=None, update_by_ss_map=None, update_by_ts_map=None, 
                 write_gpu_time=False, calc_flops=False, disable_pbar=False):

        # I don't mind bruteforce casting combination here, cuz you should worry about the meta_img size first
        idx_tuples = list(iter_product(range(self.start_pts_mesh_z.shape[0]), range(self.start_pts_mesh_z.shape[1])))

        if disable_pbar:
            pbar = idx_tuples
        elif tkinter_pbar is not None:
            pbar = tkinter_pbar(idx_tuples)
        else:
            pbar = tqdm(idx_tuples)

        accum_exec_time = 0
        accum_flops_all, accum_flops_ss, accum_flops_ts = 0, 0, 0
        for iiter, (idx_x,idx_y) in enumerate(pbar):
            zx_st, zy_st = self.start_pts_mesh_z[idx_x, idx_y]
            zx_ed = zx_st + self.config.train_params.ts_input_size 
            zy_ed = zy_st + self.config.train_params.ts_input_size

            # Handle the randomized noise input of the texture_synthesizer...
            outfeat_x_st = [start_pts_mesh[idx_x,idx_y,0] for start_pts_mesh in self.start_pts_mesh_outfeats]
            outfeat_y_st = [start_pts_mesh[idx_x,idx_y,1] for start_pts_mesh in self.start_pts_mesh_outfeats]
            outfeat_x_ed = [
                x_st + out_size for (x_st, out_size) in zip(outfeat_x_st, self.outfeat_sizes_list)]
            outfeat_y_ed = [
                y_st + out_size for (y_st, out_size) in zip(outfeat_y_st, self.outfeat_sizes_list)]
            noises = []
            for i, (fx_st, fy_st, fx_ed, fy_ed) in enumerate(zip(outfeat_x_st, outfeat_y_st, outfeat_x_ed, outfeat_y_ed)):
                # noises.append(testing_vars.noises[i][:, :, fx_st:fx_ed, fy_st:fy_ed].to(self.device))
                noises.append(self.circular_sample_width(
                    testing_vars.noises[i],
                    self.noise_widths[i],
                    fx_st,
                    fx_ed, 
                    fy_st,
                    fy_ed,
                    ).to(self.device))
            # 0:19, 0:17, 0:31, 0:29, 0:55, 0:53, 0:103, 0:101; x,y相同, 由start_pts_mesh控制
            # x相同, y为 12:31, 12:29, 24:55, 24:53, 48:103, 48:101, 96:199, 96:197  
            # noise 到 meta_height x meta_width, 和local_latent相对应, 这里偏置为12
            # local_latent的偏置为6
            # Deal with SS unfolding here
            zx_st -= self.ss_unfold_size
            zy_st -= self.ss_unfold_size
            zx_ed += self.ss_unfold_size
            zy_ed += self.ss_unfold_size
            
            # [Interactive] Decide whether the region will be updated, otherwise no need to generate
            if update_by_ss_map is not None:
                ss_cursors = zx_st, zx_ed, zy_st, zy_ed
                if not self.is_overlaping_update_map(update_by_ss_map, *ss_cursors):
                    continue
            if update_by_ts_map is not None:
                # For TS regional selection, we only select noises
                ts_cursors = outfeat_x_st[0], outfeat_x_ed[0], outfeat_y_st[0], outfeat_y_ed[0]
                if not self.is_overlaping_update_map(update_by_ts_map, *ts_cursors):
                    continue

            # cur_local_latent = testing_vars.local_latent[:, :, zx_st:zx_ed, zy_st:zy_ed].to(self.device)
            # # 0:35, 0:35; 下一循环为 0:35, 6:41
            # cur_coords = testing_vars.meta_coords[:, :, zx_st:zx_ed, zy_st:zy_ed].to(self.device)
            cur_local_latent = self.circular_sample_width(testing_vars.local_latent, testing_vars.local_latent.shape[3], zx_st, zx_ed, zy_st, zy_ed).to(self.device)
            cur_coords = self.circular_sample_width(testing_vars.meta_coords, testing_vars.meta_coords.shape[3], zx_st, zx_ed, zy_st, zy_ed).to(self.device)

            x_size = zx_ed - zx_st + 1
            y_size = zy_ed - zy_st + 1
            
            
            self.const_grid_size_x = testing_vars.meta_coords.shape[2]
            self.const_grid_size_y = testing_vars.meta_coords.shape[3]

            zy_st, circular_flag = self.get_circular_flag(zy_st, zy_ed, self.const_grid_size_y)

            if not self.attention_flag:
                print("\n\t\t\tATTENTION: Pre sampling mode is deactivate\n")
                self.attention_flag = True
            partial = 0.8
            if hasattr(self.config.train_params, "partial"):
                partial = self.config.train_params.partial
            coords_partial = {
                "p_x_st": zx_st / self.const_grid_size_x,
                "p_x_ed": (zx_st + x_size) / self.const_grid_size_x,
                "p_y_st": zy_st / self.const_grid_size_y,
                "p_y_ed": (zy_st + y_size) / self.const_grid_size_y,
                "circular_flag": circular_flag,
                "x_total": self.const_grid_size_x,
                "y_total": self.const_grid_size_y,
                "test_flag": True,
                "start_flag": iiter == 0,
                "h_step": zx_st // 6,
                "w_step": zy_st // 6,
                "y_st": zy_st,
                "y_ed": zy_ed,
                # "full_shape": self.full_shape,
                "partial": partial,
            }
            g_ema_kwargs = {
                "global_latent": testing_vars.global_latent,
                "local_latent": cur_local_latent,
                "override_coords": cur_coords,
                "coords_partial_override": coords_partial,
                "noises": noises,
                "disable_dual_latents": True,
                "calc_flops": calc_flops,
            }
            # 得到上面这些参数就可以生成小块
            """
            coords_partial = {
                    "p_x_st": x_st / self.const_grid_size_x,
                    "p_x_ed": (x_st + x_size) / self.const_grid_size_x,
                    "p_y_st": y_st / self.const_grid_size_y,
                    "p_y_ed": (y_st + y_size) / self.const_grid_size_y,
                    "circular_flag": not coords[0, 1, 1, 0] == torch.min(coords[0, 1, :, :]),
                    "x_total": self.const_grid_size_x,
                    "y_total": self.const_grid_size_y 
                }
            """
            if hasattr(testing_vars, "wplus_styles") and testing_vars.wplus_styles is not None:
                g_ema_kwargs["wplus_styles"] = testing_vars.wplus_styles
            img_x_st, img_y_st = outfeat_x_st[-1], outfeat_y_st[-1]
            img_x_ed, img_y_ed = outfeat_x_ed[-1], outfeat_y_ed[-1]
            index_tuple = (img_x_st, img_x_ed, img_y_st, img_y_ed) # 0-101, 0-101
            # 下一句生成小块
            exec_time, flops = self.maybe_parallel_inference(
                testing_vars, g_ema_kwargs=g_ema_kwargs, index_tuple=index_tuple, return_exec_time=write_gpu_time, calc_flops=calc_flops)
            accum_exec_time += exec_time
            if calc_flops:
                accum_flops_all += flops["all"]
                accum_flops_ss += flops["ss"]
                accum_flops_ts += flops["ts"]

        exec_time, flops = self.maybe_parallel_inference(
            testing_vars, flush=True, return_exec_time=write_gpu_time, calc_flops=calc_flops)
        if calc_flops:
            accum_flops_all += flops["all"]
            accum_flops_ss += flops["ss"]
            accum_flops_ts += flops["ts"]

        if write_gpu_time:
            accum_exec_time += exec_time
            print(" [*] GPU time {}".format(accum_exec_time))
            self.accum_exec_times.append(accum_exec_time)
            fmt_date = datetime.date.today().strftime("%d-%m-%Y")
            benchmark_file = os.path.join(self.save_root, "speed_benchmark_{}.txt".format(fmt_date))
            with open(benchmark_file, "a") as f:
                f.write("{:.6f}".format(accum_exec_time))

        if calc_flops:
            print(" [*] Total FLOPs: {} (SS {}, TS {})".format(
                self.pretty_print_flops(accum_flops_all), 
                self.pretty_print_flops(accum_flops_ss), 
                self.pretty_print_flops(accum_flops_ts)))

    def circular_sample_width(self, tensor:torch.Tensor, y_width, x_st, x_ed, y_st, y_ed):
        """_summary_

        Args:
            tensor (torch.Tensor): value to be sampled
            y_width (int): total_width of the tensor to be sampled
            y_st (int): start point of the slice (non-circular)
            y_ed (int): end point of the slice (non-circular: may be beyond the x_width)
        """
        if y_ed <= y_width:
            # 不超出循环的情况
            return tensor[:, :, x_st:x_ed, y_st:y_ed]
        elif y_ed <= y_width * 2:
            # 超出循环一倍以内的情况
            if y_st < y_width:
                y_ed = y_ed % y_width
                return torch.cat((tensor[:, :, x_st:x_ed, y_st:], tensor[:, :, x_st:x_ed, :y_ed]), dim=3)
            else:
                y_st = y_st % y_width
                y_ed = y_ed % y_width
                return tensor[:, :, x_st:x_ed, y_st:y_ed]
        else:
            # 超出循环两倍及以上进行递归求解
            assert y_st > y_width, "width should be larger than 35"
            y_st = y_st - y_width
            y_ed = y_ed - y_width
            return self.circular_sample_width(tensor, y_width, x_st, x_ed, y_st, y_ed)

    def save_results(self, meta_img, dump_vars=None):
        print(" [*] Saving results...")
        self.save_meta_imgs(meta_img)
        if dump_vars is not None:
            self.save_testing_vars(dump_vars)
        self.cur_global_id += self.config.train_params.batch_size

    def save_testing_vars(self, testing_vars):
        assert self.config.train_params.batch_size == 1, \
            "This is only designed to be used with the interactive tool."
        save_path = os.path.join(self.save_root, str(self.cur_global_id).zfill(6)+".pkl")
        pkl.dump(testing_vars, open(save_path, "wb"))

    def _wrap_feature(self, feat, wrap_size, dim):
        assert wrap_size < (feat.shape[dim] - 2*wrap_size), \
            "Does not expect the wrapping area is larger than a full period."
        if dim == 2:
            valid_st = feat[:, :, wrap_size:2*wrap_size]
            valid_ed = feat[:, :, -2*wrap_size:-wrap_size]
            feat[:, :, :wrap_size] = valid_ed
            feat[:, :, -wrap_size:] = valid_st
        elif dim == 3:
            valid_st = feat[:, :, :, wrap_size:2*wrap_size]
            valid_ed = feat[:, :, :, -2*wrap_size:-wrap_size]
            feat[:, :, :, :wrap_size] = valid_ed
            feat[:, :, :, -wrap_size:] = valid_st
        else:
            raise NotImplementedError(
                "I don't expect this function will be used other than spatial dims, but got {}.".format(dim))
        
    def save_meta_imgs(self, meta_img):
        self.full_image = meta_img
        # Center crop
        pad_h = (self.meta_height - self.target_height) // 2
        pad_w = (self.meta_width - self.target_width) // 2
        meta_img = meta_img[:, :, pad_h:pad_h+self.target_height, pad_w:pad_w+self.target_width]

        # Save the full image and the low-resolution image (for visualization)
        meta_img = meta_img.clamp(-1, 1).permute(0, 2, 3, 1)
        meta_img = (meta_img + 1) / 2
        meta_img_np = meta_img.numpy()
        before_id = self.cur_global_id
        for i in range(self.config.train_params.batch_size):
            global_id = self.cur_global_id + i
            save_path = os.path.join(self.save_root, str(global_id).zfill(6)+".png")
            plt.imsave(save_path, meta_img_np[i])
        after_id = global_id
        if self.save:
            save_path = os.path.join(self.latent_save_path, f"{before_id}-{after_id}.pth.tar")
            torch.save(self.save_dict, save_path)

    def _create_start_pts_mesh(self, step_size, num_steps_h, num_steps_w):
        # 定义重叠的尺寸
        start_pts_x = np.arange(num_steps_h) * step_size
        start_pts_y = np.arange(num_steps_w) * step_size
        start_pts_mesh = np.stack([
            np.repeat(start_pts_x.reshape(num_steps_h, 1), num_steps_w, axis=1),
            np.repeat(start_pts_y.reshape(1, num_steps_w), num_steps_h, axis=0),
        ], 2).astype(np.uint32) # shape: (H, W, 2)
        return start_pts_mesh

    def _init_starting_points(self):
        # 定义重叠的尺寸
        # Get all the starting points of the patches to generate, 计算生成的小块数量
        self.num_steps_h = \
            math.ceil((self.target_height - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad
        self.num_steps_w = \
            math.ceil((self.target_width  - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad

        self.start_pts_mesh_z = self._create_start_pts_mesh(
            step_size=self.latentspace_step_size, 
            num_steps_h=self.num_steps_h,
            num_steps_w=self.num_steps_w)
        self.start_pts_mesh_z += self.ss_unfold_size

        # Create this for:
        # (1) Final image pixels assignment
        # (2) Randomized noise handling within the texture synthesizer
        self.start_pts_mesh_outfeats = [
            self._create_start_pts_mesh(
                step_size=step_size,
                num_steps_h=self.num_steps_h,
                num_steps_w=self.num_steps_w,
            ) for step_size in self.outfeat_step_sizes]
        # start_pts_mesh_x = \
        #     (start_pts_mesh_z - ss_unfold_size) // latentspace_step_size * pixelspace_step_size # shape: (H, W, 2)
        # start_pts_mesh_x = start_pts_mesh_x.astype(np.uint32)

        # To avoid edge-condition on the image edge, we generate an image slightly larger than
        # requested, then center-crop to the requested resolution.
        self.meta_height = self.pixelspace_step_size * (self.num_steps_h-1) + self.outfeat_sizes_list[-1]
        self.meta_width  = self.pixelspace_step_size * (self.num_steps_w-1) + self.outfeat_sizes_list[-1]
        # height=485, width=773 (256 x 512)
        # height=485, width=965 (256 x 768)

    def compose_inv_root(self):
        return os.path.join("./logs/", self.config.var.exp_name, "test", self.config.task.prev_inv_config, "stats")

    def _init_close_loop_starting_points(self):
        # 定义重叠的尺寸
        # Get all the starting points of the patches to generate, 计算生成的小块数量
        self.num_steps_h = \
            math.ceil((self.target_height - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad
        # self.num_steps_w = \
        #     math.ceil((self.target_width  - self.outfeat_sizes_list[-1]) / self.pixelspace_step_size) + test_meta_extra_pad
        assert self.target_width % self.pixelspace_step_size == 0, f"close-loop property is not ensured\n target width {self.target_width} is not divided by target patch size {self.pixelspace_step_size}"
        self.num_steps_w_min = math.ceil(self.target_width / self.pixelspace_step_size)
        if self.target_width == 1152:
            self.num_steps_w = self.num_steps_w_min
            # for test
        else:
            self.num_steps_w = self.num_steps_w_min
        # 多出的步数为重叠回来的步数
        
        self.start_pts_mesh_z = self._create_start_pts_mesh(
            step_size=self.latentspace_step_size, 
            num_steps_h=self.num_steps_h,
            num_steps_w=self.num_steps_w)
        self.start_pts_mesh_z += self.ss_unfold_size

        # Create this for:
        # (1) Final image pixels assignment
        # (2) Randomized noise handling within the texture synthesizer
        self.start_pts_mesh_outfeats = [
            self._create_start_pts_mesh(
                step_size=step_size,
                num_steps_h=self.num_steps_h,
                num_steps_w=self.num_steps_w,
            ) for step_size in self.outfeat_step_sizes]
        # start_pts_mesh_x = \
        #     (start_pts_mesh_z - ss_unfold_size) // latentspace_step_size * pixelspace_step_size # shape: (H, W, 2)
        # start_pts_mesh_x = start_pts_mesh_x.astype(np.uint32)

        # To avoid edge-condition on the image edge, we generate an image slightly larger than
        # requested, then center-crop to the requested resolution.
        self.meta_height = self.pixelspace_step_size * (self.num_steps_h-1) + self.outfeat_sizes_list[-1]
        self.meta_width  = self.num_steps_w_min * self.pixelspace_step_size
        # height=485, width=773 (256 x 512)
        # height=485, width=965 (256 x 768)

    def get_circular_flag(self, zy_st, zy_ed, y_total):
        # 训练时可以保证zy_st不超出坐标的取值范围, 测试时无法保证
        if zy_ed > y_total:
            if zy_st < y_total:
                circular_flag = True
                return zy_st, circular_flag
            else:
                circular_flag = False
                return zy_st % y_total, circular_flag
        else:
            circular_flag = False
            return zy_st, circular_flag

