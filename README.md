# pggan-tensorflow
Tensorflow implementation of "Progressive Growing of GAN".

Please refer to the article which presents the details about algorithm and code. 

## Prerequisites

    Python 3.6
    tensorflow 2.3
    matplotlib

Usage

First,download celeba HQ dataset. 

You can download from the author's repository https://github.com/tkarras/progressive_growing_of_gans.

Or you can use CelebA-HQ MASK dataset on https://github.com/switchablenorms/CelebAMask-HQ 

Second, 
Set the proper DATA_ROOT on train.py file. 

And run "train.py"

The model will be trained for growing by **256x256** resolution. 

It took for about 3-days on a Gerforce 1080TI Graphic Card. 

`python train.py`

Each level look 800k sample and then move to next level. 

Save sample results in samples folder. Each level, save checkpoint(ckpt) in "ckpts" folder

Results

![interpolate_256](./figures/celebHQ_interpolate.gif)
