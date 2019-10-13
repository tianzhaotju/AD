from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
import os
import foolbox
import keras
import numpy as np
from keras.applications.resnet50 import ResNet50
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
X_train = X_train/255.0
X_test = X_test/255.0
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

def lenet():
    model = Sequential()
    model.add(Conv2D(input_shape=(28, 28, 1), kernel_size=(5, 5), filters=20, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
    model.add(Conv2D(kernel_size=(5, 5), filters=50, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    model = lenet()
    print('Training')
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    print('\nTesting')
    text_loss, text_accuracy = model.evaluate(X_test, y_test)

    print('\ntest loss: ', text_loss)
    print('\ntest accuracy: ', text_accuracy)

    model.save('lenet.h5')

    # instantiate model
    keras.backend.set_learning_phase(0)
    preprocessing = (np.array([0.5]), 1)
    fmodel = foolbox.models.KerasModel(model, bounds=(0, 1), preprocessing=preprocessing)

    # get source image and label
    for i in range(len(X_train)):
        image = X_train[i]
        label = int(np.argmax(y_train[i]))

        # apply attack on source image
        # ::-1 reverses the color channels, because Keras ResNet50 expects BGR instead of RGB
        attack = foolbox.attacks.FGSM(fmodel)

        adversarial = attack(image, label)

        res = int(np.argmax(model.predict(np.reshape(adversarial,(1,28,28,1)))))
        # if the attack fails, adversarial will be None and a warning will be printed
        if adversarial is None:
            print "fail attacking"
            continue
        plt.figure()

        plt.subplot(1, 3, 1)
        plt.title('Original')
        plt.imshow(1+(-1)*np.reshape(image,(28,28)),cmap= 'gray')  # division by 255 to convert [0, 255] to [0, 1]
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Adversarial')
        plt.imshow(1+(-1)*np.reshape(adversarial,(28,28)), cmap= 'gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('Difference')
        difference = adversarial - image

        #plt.imshow(np.reshape(difference,(28,28)), cmap = 'gray')
        plt.imshow(1+(-1)*np.reshape(difference, (28, 28)), cmap= 'gray')
        plt.axis('off')

        print "prediction: %d, true label: %d"%(res,label)
        plt.savefig("./picture/"+str(i)+".png")
        plt.show()
        import time
        # time.sleep(5)


        if i > 20:
            exit()
