import tensorflow as tf
from tensorflow import keras
import time

from tensorflow.python import training
import GANModel
from keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

loss = BinaryCrossentropy(from_logits=True)


def getGeneratorLoss(fake_op):
    fake_loss = loss(tf.ones_like(fake_op), fake_op)
    return fake_loss


def getDiscLoss(real_op, fake_op):
    real_loss = loss(tf.ones_like(real_op), real_op)
    fake_loss = loss(tf.zeros_like(fake_op), fake_op)
    total_loss = fake_loss + real_loss
    return total_loss


def getGeneratorOptimizer(LR=3e-4):
    return Adam(learning_rate=LR)


def getDiscriminatorOptimizer(LR=3e-4):
    return Adam(learning_rate=LR)


class TrainGan:
    def train(self, dataset, epochs, BATCH_SIZE, NOISE_DIM, discriminator, generator, get_loss=False, lr=3e-4):
        DISC_OBJ = GANModel.Discriminator()
        GEN_OBJ = GANModel.Generator()
        GEN_LOSS = []
        DISC_LOSS = []
        for epoch in range(epochs):
            print("Starting Training for epoch:", epoch)
            start = time.time()

            for image_batch in dataset:
                if get_loss == True:
                    gen_loss, disc_loss = self.train_step(
                        image_batch, BATCH_SIZE, NOISE_DIM, discriminator, generator, get_loss, lr)
                    GEN_LOSS.append(gen_loss)
                    DISC_LOSS.append(disc_loss)
                else:
                    self.train_step(image_batch, BATCH_SIZE, NOISE_DIM,
                                    discriminator, generator, get_loss)

            end = time.time()
            print("Time Taken for Epoch:", epoch+1,
                  " is:", (str)(end-start), " sec")
        return GEN_LOSS, DISC_LOSS

    def train_step(self, images, BATCH_SIZE, NOISE_DIM, discriminator, generator, get_loss, lr):
        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_op = discriminator(images, training=True)
            fake_op = discriminator(generated_images, training=True)

            # Determining the losses
            gen_loss = getGeneratorLoss(fake_op)
            disc_loss = getDiscLoss(real_op, fake_op)

        # Getting the gradients of the models based on losses
        gradients_of_generator = gen_tape.gradient(
            gen_loss, generator.trainable_variables)
        gradients_of_disc = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables)

        # Applying the gradients for training the network
        getGeneratorOptimizer(lr).apply_gradients(
            zip(gradients_of_generator, generator.trainable_variables))
        getDiscriminatorOptimizer(lr).apply_gradients(
            zip(gradients_of_disc, discriminator.trainable_variables))

        if get_loss == True:
            return gen_loss, disc_loss
