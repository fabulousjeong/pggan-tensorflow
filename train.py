import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot
from math import sqrt
from PIL import Image
import os

from pgan import PGAN, WeightedSum
from tensorflow.keras import backend

# Create a Keras callback that periodically saves generated images and updates alpha in WeightedSum layers
class GANMonitor(keras.callbacks.Callback):
  def __init__(self, num_img=16, latent_dim=512, prefix=''):
    self.num_img = num_img
    self.latent_dim = latent_dim
    self.random_latent_vectors = tf.random.normal(shape=[num_img, self.latent_dim], seed=9434)
    self.steps_per_epoch = 0
    self.epochs = 0
    self.steps = self.steps_per_epoch * self.epochs
    self.n_epoch = 0
    self.prefix = prefix
  
  def set_prefix(self, prefix=''):
    self.prefix = prefix
  
  def set_steps(self, steps_per_epoch, epochs):
    self.steps_per_epoch = steps_per_epoch
    self.epochs = epochs
    self.steps = self.steps_per_epoch * self.epochs

  def on_epoch_begin(self, epoch, logs=None):
    self.n_epoch = epoch


  def on_epoch_end(self, epoch, logs=None):
    samples = self.model.generator(self.random_latent_vectors)
    samples = (samples * 0.5) + 0.5
    n_grid = int(sqrt(self.num_img))

    fig, axes = pyplot.subplots(n_grid, n_grid, figsize=(4*n_grid, 4*n_grid))
    sample_grid = np.reshape(samples[:n_grid * n_grid], (n_grid, n_grid, samples.shape[1], samples.shape[2], samples.shape[3]))

    for i in range(n_grid):
      for j in range(n_grid):
        axes[i][j].set_axis_off()
        samples_grid_i_j = Image.fromarray((sample_grid[i][j] * 255).astype(np.uint8))
        samples_grid_i_j = samples_grid_i_j.resize((128,128))
        axes[i][j].imshow(np.array(samples_grid_i_j))
    title = f'images/plot_{self.prefix}_{epoch:05d}.png'
    pyplot.savefig(title, bbox_inches='tight')
    print(f'\n saved {title}')
    pyplot.close(fig)
  

  def on_batch_begin(self, batch, logs=None):
    # Update alpha in WeightedSum layers
    alpha = ((self.n_epoch * self.steps_per_epoch) + batch) / float(self.steps - 1)
    #print(f'\n {self.steps}, {self.n_epoch}, {self.steps_per_epoch}, {alpha}')
    for layer in self.model.generator.layers:
      if isinstance(layer, WeightedSum):
        backend.set_value(layer.alpha, alpha)
    for layer in self.model.discriminator.layers:
      if isinstance(layer, WeightedSum):
        backend.set_value(layer.alpha, alpha)


# DEFINE FILEPATH AND PARAMETERS
# can use celeb A mask dataset on https://github.com/switchablenorms/CelebAMask-HQ 
DATA_ROOT = './CelebAMask-HQ' 
NOISE_DIM = 512
# Set the number of batches, epochs and steps for trainining.
# Look 800k images(16x50x1000) per each lavel
BATCH_SIZE = [16, 16, 16, 16, 16, 16, 4]
EPOCHS = 50
STEPS_PER_EPOCH = 1000





# Normalilze [-1, 1] input images
def preprocessing_image(img):
  img = img.astype('float32')
  img = (img - 127.5) / 127.5
  return img

train_image_generator = ImageDataGenerator(horizontal_flip=True, preprocessing_function=preprocessing_image)
train_dataset = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE[0],
                                                          directory=DATA_ROOT,
                                                          shuffle=True,
                                                          target_size=(4,4),
                                                          class_mode='binary')

# Instantiate the optimizer for both networks
# learning_rate will be equalized per each layers by the WeightScaling scheme
generator_optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)



cbk = GANMonitor(num_img=64, latent_dim=NOISE_DIM, prefix='0_init')
cbk.set_steps(steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS)

# Instantiate the PGAN(PG-GAN) model.
pgan = PGAN(
    latent_dim = NOISE_DIM, 
    d_steps = 1,
)

checkpoint_path = f"ckpts/pgan_{cbk.prefix}.ckpt"

# Compile models
pgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
)

print(len(train_dataset))
# Draw models
tf.keras.utils.plot_model(pgan.generator, to_file=f'generator_{pgan.n_depth}.png', show_shapes=True)
tf.keras.utils.plot_model(pgan.discriminator, to_file=f'discriminator_{pgan.n_depth}.png', show_shapes=True)

# Start training the initial generator and discriminator
pgan.fit(train_dataset, steps_per_epoch = STEPS_PER_EPOCH, epochs = EPOCHS, callbacks=[cbk])
pgan.save_weights(checkpoint_path)

# Train faded-in / stabilized generators and discriminators
for n_depth in range(1, 7):
  # Set current level(depth)
  pgan.n_depth = n_depth

  # Set parameters like epochs, steps, batch size and image size
  steps_per_epoch = STEPS_PER_EPOCH
  epochs = int(EPOCHS*(BATCH_SIZE[0]/BATCH_SIZE[n_depth]))
  DATA_ROOT = f'/home/munan/DB/celebHQ/CelebAMask-HQ/CelebA-HQ-{n_depth}'
  train_dataset = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE[n_depth],
                                                          directory=DATA_ROOT,
                                                          shuffle=True,
                                                          target_size=(4*(2**n_depth), 4*(2**n_depth)),
                                                          class_mode='binary')
  cbk.set_prefix(prefix=f'{n_depth}_fade_in')
  cbk.set_steps(steps_per_epoch=steps_per_epoch, epochs=epochs)

  # Put fade in generator and discriminator
  pgan.fade_in_generator()
  pgan.fade_in_discriminator()

  # Draw fade in generator and discriminator
  tf.keras.utils.plot_model(pgan.generator, to_file=f'generator_{n_depth}_fade_in.png', show_shapes=True)
  tf.keras.utils.plot_model(pgan.discriminator, to_file=f'discriminator_{n_depth}_fade_in.png', show_shapes=True)

  pgan.compile(
      d_optimizer=discriminator_optimizer,
      g_optimizer=generator_optimizer,
  )
  # Train fade in generator and discriminator
  pgan.fit(train_dataset, steps_per_epoch = steps_per_epoch, epochs = epochs, callbacks=[cbk])
  # Save models
  checkpoint_path = f"ckpts/pgan_{cbk.prefix}.ckpt"
  pgan.save_weights(checkpoint_path)

  # Change to stabilized generator and discriminator
  cbk.set_prefix(prefix=f'{n_depth}_stabilize')
  pgan.stabilize_generator()
  pgan.stabilize_discriminator()

  # Draw stabilized generator and discriminator
  tf.keras.utils.plot_model(pgan.generator, to_file=f'generator_{n_depth}_stabilize.png', show_shapes=True)
  tf.keras.utils.plot_model(pgan.discriminator, to_file=f'discriminator_{n_depth}_stabilize.png', show_shapes=True)
  pgan.compile(
      d_optimizer=discriminator_optimizer,
      g_optimizer=generator_optimizer,
  )
  # Train stabilized generator and discriminator
  pgan.fit(train_dataset, steps_per_epoch = steps_per_epoch, epochs = epochs, callbacks=[cbk])
  # Save models
  checkpoint_path = f"ckpts/pgan_{cbk.prefix}.ckpt"
  pgan.save_weights(checkpoint_path)

