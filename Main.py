from os import name
import GANModel
import TrainGAN
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from Preprocessing import Preprocess

'''
  How to test?
  -> Go to your google colab notebook
  -> Upload the files GANModel.py and TrainGAN.py on your current session
  -> Now Copy the code for Main.py in your notebook and run your cell
    ** Make sure you select GPU for your runtime as that will speed up the training
  -> The Results will be shown after the training is complete
'''


# Defining the parameters
BATCH_SIZE = 256
EPOCHS = 5
DISC_BASE_FILTERS = 64
DISC_KERNEL_SIZE = (5, 5)
INPUT_SHAPE = (28, 28)
OUTPUT_CHANNELS = 1
BUFFER_SIZE = 60000
GEN_BASE_FILTERS = 32
GEN_KERNEL_SIZE = (5, 5)
NOISE_DIM = 100
SHOW_LOSS_PLOT = True
SHOW_SAMPLES = True
SHOW_SAMPLES_SIZE = 10
LEARNING_RATE = 3e-4

# Defining the model
DISC = GANModel.Discriminator().getDiscriminator(INPUT_SHAPE=INPUT_SHAPE,
                                                 BASE_FILTERS=DISC_BASE_FILTERS, KERNEL_SIZE=DISC_KERNEL_SIZE)
GEN = GANModel.Generator().getGenerator(
    BASE_FILTERS=GEN_BASE_FILTERS, KERNEL_SIZE=GEN_KERNEL_SIZE)


# Initializing preprocessing object
preprocess = Preprocess()

# Loading the data
(train_images, _), (test_images, _) = mnist.load_data()

# Reshaping Images
preprocess.reshape_images(train_images, INPUT_SHAPE, OUTPUT_CHANNELS)
preprocess.convert_to_slices(BUFFER_SIZE, BATCH_SIZE)
train_dataset = preprocess.get_all_images()

# Getting generator and discriminator loss
GEN_LOSS, DISC_LOSS = TrainGAN.TrainGan().train(
    train_dataset, EPOCHS, BATCH_SIZE, NOISE_DIM, DISC, GEN, True)

# Visualizing the data
if SHOW_SAMPLES == True:
    noise = tf.random.normal([256, NOISE_DIM])
    gen_image = GEN(noise, training=False)
    for i in range(SHOW_SAMPLES_SIZE):
        plt.imshow(gen_image[i].numpy().reshape(28, 28))
        plt.show()
if SHOW_LOSS_PLOT == True:
    plt.plot(GEN_LOSS, label="Generator Loss")
    plt.plot(DISC_LOSS, label="Discriminator Loss")
    plt.legend()
    plt.show()
