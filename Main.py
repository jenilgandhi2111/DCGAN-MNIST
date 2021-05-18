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

# Defining the model
DISC = GANModel.Discriminator().makeDiscriminator(INPUT_SHAPE=INPUT_SHAPE,BASE_FILTERS=DISC_BASE_FILTERS,KERNEL_SIZE=DISC_KERNEL_SIZE)
GEN = GANModel.Generator().makeGenerator(BASE_FILTERS=GEN_BASE_FILTERS,KERNEL_SIZE=GEN_KERNEL_SIZE)


(train_images,_),(test_images,_) = mnist.load_data()
print("1.) Starting reshaping of images")
train_images = train_images.reshape(train_images.shape[0],28,28,1)
# Normalizing images from [-1,1]
train_images=(train_images-127.5)/127.5
print("2.) Reshaping of images Completed..")


# Making images in batches'
print("3.) Making train dataset")
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
print("4.) Done making train dataset")


print("5.) Training the Gan started")
GEN_LOSS,DISC_LOSS = TrainGAN.TrainGan().train(train_dataset,EPOCHS,BATCH_SIZE,NOISE_DIM,DISC,GEN,True)
print("6.) Training the Gan Finished")


print("5.) Visualizing the Gan")
# noise = tf.random.normal([256, NOISE_DIM])
# gen_image=GEN(noise,training=False)
# for i in range(20):
#   plt.imshow(gen_image[i].numpy().reshape(28,28))
#   plt.show()
plt.plot(GEN_LOSS,label="Generator Loss")
plt.plot(DISC_LOSS,label="Discriminator Loss")
plt.legend()
plt.show()

print("7.) Done all steps")

