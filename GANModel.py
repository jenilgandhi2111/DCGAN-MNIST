from os import name
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, Conv2DTranspose, MaxPool2D, LeakyReLU, BatchNormalization, Reshape, Flatten, Dropout
from keras.models import Sequential
from keras.losses import BinaryCrossentropy


class Generator:
    def getGenerator(self, BASE_FILTERS=32, KERNEL_SIZE=(5, 5), FILTERS_CONV=[128, 128], OUTPUT_CHANNELS=1):

        self.loss = BinaryCrossentropy(from_logits=True)

        # Defining the model and Initializing the input layer
        self.generator = Sequential(name="generator")
        # Layer 1 for generator
        self.generator.add(Dense(7*7*256, use_bias=False, input_shape=(100,)))
        self.generator.add(BatchNormalization(momentum=0.9))
        self.generator.add(LeakyReLU())

        # layer 2 for generator
        self.generator.add(Reshape((7, 7, 256)))
        self.generator.add(Conv2DTranspose(FILTERS_CONV[0], kernel_size=(
            5, 5), strides=(1, 1), padding="same", use_bias=False))
        self.generator.add(BatchNormalization())
        self.generator.add(LeakyReLU())

        # Layer 3 for generator
        self.generator.add(Conv2DTranspose(FILTERS_CONV[1], kernel_size=(
            5, 5), strides=(2, 2), padding="same", use_bias=False))
        self.generator.add(BatchNormalization())
        self.generator.add(LeakyReLU())

        # Output layer
        self.generator.add(Conv2DTranspose(OUTPUT_CHANNELS, kernel_size=(
            5, 5), strides=(2, 2), padding="same", use_bias=False, activation="tanh"))
        # The above layer would generate an upsampled image form the random nosie it was given

        return self.generator

    def getGeneratorModelSummary(self):
        return self.Gself.enerator.summary()


class Discriminator:
    def getDiscriminator(self, INPUT_SHAPE=(28, 28, 1), BASE_FILTERS=64, KERNEL_SIZE=(5, 5)):

        self.loss = BinaryCrossentropy()

        # Defining the model
        self.discriminator = Sequential(name="Discriminator")

        # Input Layer
        self.discriminator.add(Conv2D(
            BASE_FILTERS, kernel_size=KERNEL_SIZE, use_bias=False, strides=(2, 2), padding="same"))
        self.discriminator.add(LeakyReLU())
        self.discriminator.add(Dropout(0.3))

        # Layer 1
        self.discriminator.add(Conv2D(
            BASE_FILTERS*2, kernel_size=KERNEL_SIZE, strides=(2, 2), padding="same", use_bias=False))
        self.discriminator.add(LeakyReLU())
        self.discriminator.add(Dropout(0.2))

        # Layer2 predicts whetheer the ouptut is real or fake
        self.discriminator.add(Flatten())
        self.discriminator.add(Dense(1))

        return self.discriminator

    def getDiscrminatorModelSummary(self):
        return self.Discriminator.summary()
