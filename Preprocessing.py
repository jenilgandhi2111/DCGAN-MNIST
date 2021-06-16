import cv2
from numpy.core.fromnumeric import resize
import tensorflow as tf
import numpy as np
from tqdm import tqdm


class ImagePreProcess:
    def __init__(self):
        pass

    def resize_image(self, image, resize_to, is_rgb=1):

        if is_rgb == 1:
            image = cv2.resize(image, resize_to)
            image = np.array(image).reshape(resize_to[0], resize_to[1], 1)
            return image
        else:
            image = cv2.resize(image, resize_to)
            image = np.array(image).reshape(resize_to[0], resize_to[1], 3)
            return image


class Preprocess:
    def __init__(self):
        pass

    def reshape_images(self, images_array, reshape_to=(28, 28), output_channels=1):
        imagePreProcess = ImagePreProcess()
        '''
        > Input params: images_array( all images as a numpy arraay ) , reshape_to = (resize size),
                        output_channels = (1-> If Grayscale image , 3-> If Color image)
        > Reshape images function resizes images to 28*28*1 by default but
          you could provide your own reshape size and output channels
        '''
        self.all_images = np.array(images_array)

        try:
            # if(self.all_images[0].shape != reshape_to):
            tp = []
            print("> Resizing images to :")
            print(reshape_to)
            for image in self.all_images:
                tp.append(imagePreProcess.resize_image(
                    image, reshape_to, output_channels))
            self.all_images = np.array(tp)
            print("> Started Reshaping of images")

            # Normalizing to -1 to 1
            self.all_images = self.all_images-127.5/127.5
            self.all_images = self.all_images.reshape(
                self.all_images.shape[0], reshape_to[0], reshape_to[1], output_channels)
        except Exception as e:
            raise e

    def convert_to_slices(self, buffer_size, batch_size):
        print("> Converting to slices")
        print(np.array(self.all_images).shape)
        self.all_images = np.array(self.all_images, dtype='bfloat16')
        self.all_images = tf.data.Dataset.from_tensor_slices(
            self.all_images).shuffle(buffer_size).batch(batch_size)

    def get_all_images(self):
        return self.all_images
