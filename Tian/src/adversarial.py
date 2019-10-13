import foolbox
import keras
import numpy as np
from keras.applications.resnet50 import ResNet50
import matplotlib.pyplot as plt
import time

def adversarial(nnet,images,labels):
    # instantiate model
    keras.backend.set_learning_phase(0)
    preprocessing = (np.array([104]), 1)

    fmodel = foolbox.models.TheanoModel(nnet.input_layer_tensor, nnet.logits, (0,1), 2, 1, preprocessing)

    # apply attack on source image
    # ::-1 reverses the color channels, because Keras ResNet50 expects BGR instead of RGB
    print np.shape(images)
    attack = foolbox.attacks.FGSM(fmodel)
    print 'starting attack'
    for i in range(len(images)):
        image = images[i][0]
        label = labels[i]
        adversarial = attack(np.reshape(image,(28,28,1)), label)
        print label
        print np.shape(images)
        #plt.plot(image)
        print np.shape(image)
        print image
        plt.show()
        # adversarial = attack(image[:, :, ::-1], label)

        # if the attack fails, adversarial will be None and a warning will be printed
        print adversarial
        time.sleep(2)
        print 'ending attack'

        plt.figure()

        plt.subplot(1, 3, 1)
        plt.title('Original')
        plt.imshow(image)  # division by 255 to convert [0, 255] to [0, 1]
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Adversarial')
        plt.imshow(adversarial)  # ::-1 to convert BGR to RGB
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('Difference')
        difference = adversarial - image
        plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
        plt.axis('off')

        plt.show()


if __name__ == '__main__':
    nnet = NNet("MNIST")
    adversarial(nnet)