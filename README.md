# pggan-tensorflow
Tensorflow implementation of "[Progressive Growing of GAN](https://arxiv.org/abs/1710.10196)".
Please refer to the paper which presents the details about algorithm. 

For more information on the code, please refer to the following **[Medium Story Link](https://fabulousjeong.medium.com/tensorflow2-0-pggan-progressive-growing-of-gans-for-improved-quality-stability-and-variation-67a474b39356)**


## Prerequisites

    Python 3.6
    tensorflow 2.3
    matplotlib

## Usage

**1.**

download celeba HQ dataset. 

You can download from the author's repository https://github.com/tkarras/progressive_growing_of_gans.

Or you can use CelebA-HQ MASK dataset on https://github.com/switchablenorms/CelebAMask-HQ 


**2.** 

Set the proper path to define the **DATA_ROOT** on train.py file. 


**3.**

run "train.py"

The model will be trained for growing by **256x256** resolution. 

It took for about 3-days on a Gerforce 1080TI Graphic Card. 

`python train.py`

Each level look 800k sample and then move to next level. 

Save sample results in samples folder. Each level, save checkpoint(ckpt) in "ckpts" folder


## Results

![interpolate_256](./figures/celebHQ_interpolate.gif)


## Acknowledges

TKarras's repository, https://github.com/tkarras/progressive_growing_of_gans

Keras WGAN-GP Example, https://keras.io/examples/generative/wgan_gp/ 
