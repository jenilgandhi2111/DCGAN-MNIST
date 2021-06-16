# DCGAN-MNIST
```The DCGAN(Deep Convolutional Generative Adverserial Network)provided some random noise is used to generate new samples from a given data of some sample images , for this demonstration I have used MNIST data set for numbers```
``We could evidently see that how training for more epochs let's us improve more and more below are some sample images I generated from the Generator``
<br>
**Less trained GAN example**
<kbd>
<img src="https://github.com/jenilgandhi2111/DCGAN-MNIST/blob/main/Output%20Examples/LessTrainedExample.png"/><br>
</kbd>
<br>
**After training for 25 epochs**
<kbd>
<img src="https://github.com/jenilgandhi2111/DCGAN-MNIST/blob/main/Output%20Examples/AllNumberMoreTrained.png"/>
</kbd>
<br>

## How Gan's Work:
* The Gan's were introduced by Ian Goodfellow in 2014. Generative adversarial networks (GANs) are algorithmic architectures that use two neural networks, pitting one against the other (thus the “adversarial”) in order to generate new, synthetic instances of data that can pass for real data.

* It is more like a chor police game or a minimax game where the chor tries to counterfeit notes and everytime the chor makes a mistake it is penalized and it tries to get better and better on counterfeiting the notes and the police tries to catch the chor evrytime and the police gets better and better on catching the chor . At some point of time the police would not be able to tell the difference between real and fake notes. 

* Discrminator Loss: △ log( Discriminator(Real image) ) + ( 1 - Discriminator(Generated Image) )<br>
Generator Loss: ▽ log(1-Discrminator(Generated image))<br>
(Note: △ symbol means maximize and ▽ symbol means minimize )

* Below given is a architectural view of GAN.<br><kbd><img src="https://miro.medium.com/max/601/1*Y_AGVp0EEGEpB1Q25G6edQ.jpeg"/></kbd>

<b>In the above model we need to tune the hyperparameters to use it . I have handpicked some of the best results I got</b>
                
