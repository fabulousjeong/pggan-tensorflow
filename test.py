import numpy as np
import tensorflow as tf
from tensorflow import keras

from matplotlib import pyplot
from math import sqrt
from PIL import Image
import os

from pgan import PGAN
from tensorflow.keras import backend

def saveSample(generator, random_latent_vectors, prefix):
  samples = generator(random_latent_vectors)
  samples = (samples * 0.5) + 0.5
  n_grid = int(sqrt(random_latent_vectors.shape[0]))

  fig, axes = pyplot.subplots(n_grid, n_grid, figsize=(8*n_grid, 8*n_grid))
  sample_grid = np.reshape(samples[:n_grid * n_grid], (n_grid, n_grid, samples.shape[1], samples.shape[2], samples.shape[3]))

  for i in range(n_grid):
    for j in range(n_grid):
      axes[i][j].set_axis_off()
      samples_grid_i_j = Image.fromarray((sample_grid[i][j] * 255).astype(np.uint8))
      samples_grid_i_j = samples_grid_i_j.resize((256,256))
      axes[i][j].imshow(np.array(samples_grid_i_j))
  title = f'test/plot_{prefix}_{0:05d}.png'
  pyplot.savefig(title, bbox_inches='tight')
  print(f'\n saved {title}')
  pyplot.close(fig)




NOISE_DIM = 512
NUM_SAMPLE = 4
random_latent_vectors = tf.random.normal(shape=[NUM_SAMPLE, NOISE_DIM])#, seed=9434)

# Instantiate the PGAN(PG-GAN) model.
pgan = PGAN(
    latent_dim = NOISE_DIM, 
    d_steps = 1,
)

# Load weight and generate samples per each level. 
prefix='0_init'
pgan.load_weights(f"ckpts/pgan_{prefix}.ckpt")
saveSample(pgan.generator, random_latent_vectors, prefix)

#inference
for n_depth in range(1,7):
  pgan.n_depth = n_depth
  prefix=f'{n_depth}_fade_in'
  pgan.fade_in_generator()
  pgan.fade_in_discriminator()

  pgan.load_weights(f"ckpts/pgan_{prefix}.ckpt")
  saveSample(pgan.generator, random_latent_vectors, prefix)

  prefix=f'{n_depth}_stabilize'
  pgan.stabilize_generator()
  pgan.stabilize_discriminator()

  pgan.load_weights(f"ckpts/pgan_{prefix}.ckpt")
  saveSample(pgan.generator, random_latent_vectors, prefix)
pgan.load_weights(f"ckpts/pgan_{prefix}.ckpt")


# Generate interpolated samples. 
sample_0 = tf.random.normal(shape=[NUM_SAMPLE, NOISE_DIM], seed = 4311)
sample_1 = tf.random.normal(shape=[NUM_SAMPLE, NOISE_DIM], seed = 55)

steps = 120
dt = (sample_0 - sample_1)/steps
for i in range(1, steps+1):
    random_latent_vectors = sample_0 + dt*i
    print(random_latent_vectors[:4])
    prefix = f'{i:03d}'
    saveSample(pgan.generator, random_latent_vectors, prefix)

