import tensorflow as tf
from tensorflow import keras
import time
import GANModel
from keras.losses import BinaryCrossentropy

loss = BinaryCrossentropy(from_logits=True)
def getGeneratorLoss(fake_op):
        fake_loss = loss(tf.ones_like(fake_op),fake_op)
        return fake_loss
def getDiscLoss(real_op,fake_op):
        real_loss = loss(tf.ones_like(real_op),real_op)
        fake_loss = loss(tf.zeros_like(fake_op),fake_op)
        total_loss = fake_loss + real_loss
        return total_loss

        
class TrainGan:
    def __init__(self,dataset,epochs,BATCH_SIZE,NOISE_DIM,discriminator,generator):
        DISC_OBJ = GANModel.Discriminator()
        GEN_OBJ = GANModel.Generator()
        for epoch in range(epochs):
            print("Starting Training for epoch:",epoch)
            start = time.time

            for image_batch in dataset:
                self.train_step(image_batch,BATCH_SIZE,NOISE_DIM,discriminator,generator,DISC_OBJ,GEN_OBJ)
            
            end = time.time
            # print("Time taken:",(str)(end-start)," Seconds")
    
    def train_step(self,images,BATCH_SIZE,NOISE_DIM,discriminator,generator,disc_obj,gen_obj):
        noise = tf.random.normal([BATCH_SIZE,NOISE_DIM])
        
        with tf.GradientTape() as gen_tape,tf.GradientTape() as disc_tape:
            generated_images = generator(noise)

            real_op = discriminator(images,training = True)
            fake_op = discriminator(generated_images,training = True)

            # Determining the losses
            gen_loss = getGeneratorLoss(fake_op)
            disc_loss = getDiscLoss(real_op,fake_op)

            #Getting the gradients of the models based on losses
        gradients_of_generator = gen_tape.gradient(gen_loss,generator.trainable_variables)
        gradients_of_disc = disc_tape.gradient(disc_loss,discriminator.trainable_variables)
                
            #Applying the gradients for training the network
        gen_obj.getGeneratorOptimizer().apply_gradients(zip(gradients_of_generator,generator.trainable_variables))
        disc_obj.getDiscriminatorOptimizer().apply_gradients(zip(gradients_of_disc,discriminator.trainable_variables))

        