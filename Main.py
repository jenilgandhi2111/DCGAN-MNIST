from os import name
import GANModel
import TrainGAN
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Defining the parameters
BATCH_SIZE = 256
EPOCHS = 1
DISC_BASE_FILTERS = 64
DISC_KERNEL_SIZE = (5,5)
INPUT_SHAPE = (28,28,1)
BUFFER_SIZE = 60000
GEN_BASE_FILTERS = 32
GEN_KERNEL_SIZE = (5,5)
NOISE_DIM = 100
SHOW_LOSS_PLOT = True
SHOW_SAMPLES = True
SHOW_SAMPLES_SIZE=10

# Defining the model
DISC = GANModel.Discriminator().getDiscriminator(INPUT_SHAPE=INPUT_SHAPE,BASE_FILTERS=DISC_BASE_FILTERS,KERNEL_SIZE=DISC_KERNEL_SIZE)
GEN = GANModel.Generator().getGenerator(BASE_FILTERS=GEN_BASE_FILTERS,KERNEL_SIZE=GEN_KERNEL_SIZE)

# Loading the data
(train_images,_),(test_images,_) = mnist.load_data()
train_images = train_images.reshape(train_images.shape[0],28,28,1)

# Normalizing images from [-1,1]
train_images=(train_images-127.5)/127.5

# Making images in batches'
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Getting generator and discriminator loss
GEN_LOSS,DISC_LOSS = TrainGAN.TrainGan().train(train_dataset,EPOCHS,BATCH_SIZE,NOISE_DIM,DISC,GEN,True)

# Visualizing the data
if SHOW_SAMPLES == True:
  noise = tf.random.normal([256, NOISE_DIM])
  gen_image=GEN(noise,training=False)
  for i in range(SHOW_SAMPLES_SIZE):
    plt.imshow(gen_image[i].numpy().reshape(28,28))
    plt.show()
if SHOW_LOSS_PLOT == True:
  plt.plot(GEN_LOSS,label="Generator Loss")
  plt.plot(DISC_LOSS,label="Discriminator Loss")
  plt.legend()
  plt.show()



