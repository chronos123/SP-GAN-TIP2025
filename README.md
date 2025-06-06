## Environmets
```sh
conda env create -f configs/env/environment.yml
```

## Dataset
Download the matterport3D dataset from "https://niessner.github.io/Matterport/"

Run the following command to get panoramic images for training
```sh
python gen_pano_dataset.py
```

Run the following command to construct the lmdb database for training and testing

## Pre-trained Weights

Get the pretrained weights from "https://drive.google.com/file/d/1pNZbIyBcSrfQZv3Hr_whzkGOAp72S-fY/view?usp=drive_link"

Download the 'SP-GAN.ckpt' and put it at 'weights/SP-GAN.ckpt'

## Training

```sh
python train.py configs/model/spgan.yaml
```

## Testing

```sh
CUDA_VISIBLE_DEVICES=0 python test.py --model-config configs/model/spgan.yaml --test-config configs/test/spgan_384x768.yaml --ckpt weights/SP-GAN.ckpt
```


