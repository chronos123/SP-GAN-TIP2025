## Matterport3d cube2sphere
# Prerequirements:
# 1. Download the "skybox" category of Matterport3d dataset.
# 2. Install cube2sphere library

import os
import zipfile

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, Lock

import matplotlib.pyplot as plt

width = 768
height = 384
edge_cutoff_ratio = 0.6667


matterport_dataset_path = './data/matterport3d/train/v1/scans/'
panorama_output_path = f'./data/matterport3d_{width}x{height}/full_pano/'
clipped_panorama_output_path = f'./data_new/matterport3d_panorama_{width}x{int(height*edge_cutoff_ratio)}'

cube2sphere_output_shape = (width, int(height))

final_shape = (width, round(height * edge_cutoff_ratio)) 

clip_len = int((height - round(height * edge_cutoff_ratio)) / 2)

if not os.path.exists(panorama_output_path):
    os.makedirs(panorama_output_path)
if not os.path.exists(clipped_panorama_output_path):
    os.makedirs(clipped_panorama_output_path)

def unzip(zip_path, to_path):
    zip_ref = zipfile.ZipFile(zip_path, 'r')
    zip_ref.extractall(to_path)
    zip_ref.close()

all_house_dirs = glob(matterport_dataset_path + '*')

all_panorama_imgs = []
all_panorama_ids = {} 

def run_cmd(*args):
    os.system("".join(args))

# Allocate all panorama images
for d in tqdm(all_house_dirs):
    
    house_id = d.split("/")[-1]
    
    # unzip file if hasn't yet
    if not os.path.exists(d + '/' + house_id):
        print("!", house_id)
        unzip(d + '/matterport_skybox_images.zip', to_path=d)
    
    skybox_data_path = "{}/{}/matterport_skybox_images/".format(d, house_id)
    for name in os.listdir(skybox_data_path):
        if 'jpg' in name:
            panorama_id = name[:-4].split('_')[0]
            all_panorama_ids[panorama_id] = skybox_data_path

skybox_idx = {
    'top': 0,
    'back': 1,
    'right': 2,
    'front': 3,
    'left': 4,
    'bottom': 5,
}

def compose_skybox_path(position, panorama_id, panorama_path):
    pos_idx = skybox_idx[position]
    return '{}/{}_skybox{}_sami.jpg'.format(panorama_path, panorama_id, pos_idx)

tasks = []
for panorama_id in tqdm(all_panorama_ids):
    path = all_panorama_ids[panorama_id]

    # -b <blender_path>
    command = ' '.join([
        "cube2sphere",
        compose_skybox_path('front', panorama_id, path),
        compose_skybox_path('back', panorama_id, path),
        compose_skybox_path('right', panorama_id, path),
        compose_skybox_path('left', panorama_id, path),
        compose_skybox_path('top', panorama_id, path),
        compose_skybox_path('bottom', panorama_id, path),
        "-r {} {}".format(cube2sphere_output_shape[0], cube2sphere_output_shape[1]),
        "-f PNG",
        "-o {}/{}".format(panorama_output_path, panorama_id)])
    tasks.append(command)

pool = Pool(20)
print(f"len tasks is {len(tasks)}")
print(tasks[0])
pool.starmap(run_cmd, tasks)

def clip_img_margin(img, clip_len=64):
    HH, WW, _ = img.shape
    if final_shape[0]==HH and final_shape[1]==WW:
        return img
    else:
        return img[clip_len:HH-clip_len, :, :]


lock = Lock()
def clip_pano(panorama_id):
    img = plt.imread('{}/{}0001.png'.format(panorama_output_path, panorama_id))
    clipped_img = clip_img_margin(img, clip_len=clip_len)
    output_path = '{}/{}.png'.format(clipped_panorama_output_path, panorama_id)
    plt.imsave(output_path, clipped_img)

pool = Pool(60)
print(f"len tasks is {len(all_panorama_ids)}")
pool.starmap(clip_pano, [[e] for e in all_panorama_ids])
