import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import config as cfg
import os
from LeNet import Lenet
import foolbox
import keras
import matplotlib.pyplot as plt
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    sess = tf.InteractiveSession()
    batch_size = cfg.BATCH_SIZE
    parameter_path = cfg.PARAMETER_FILE
    lenet = Lenet()
    max_iter = 150000

    saver = tf.train.Saver()
    if os.path.exists(parameter_path):
        saver.restore(parameter_path)
    else:
        sess.run(tf.initialize_all_variables())

    for i in range(max_iter):
        batch = mnist.train.next_batch(500)
        if i % 100 == 0:
            [train_accuracy, loss] = sess.run([lenet.train_accuracy, lenet.loss], feed_dict={
                lenet.raw_input_image: batch[0], lenet.raw_input_label: batch[1]
            })

            [_,train_accuracy,loss] = sess.run([lenet.train_op, lenet.train_accuracy, lenet.loss],feed_dict={lenet.raw_input_image: batch[0],lenet.raw_input_label: batch[1]
            })
            print("step %d, training accuracy %g %g" % (i, train_accuracy,loss))

    save_path = saver.save(sess, parameter_path)
    batch = mnist.train.next_batch(1000)
    # adversarial(lenet,batch[0],batch[1])

    images = batch[0]
    labels = batch[1]


# def adversarial(nnet,images,labels):
    # instantiate model

    preprocessing = (np.array([0.5]), 1)
    fmodel = foolbox.models.TensorFlowModel(lenet.raw_input_image, lenet.train_digits, (0, 1), 1, preprocessing)

    #print np.argmax(fmodel.forward_one(images))

    # apply attack on source image
    # ::-1 reverses the color channels, because Keras ResNet50 expects BGR instead of RGB
    criterion = foolbox.criteria.Misclassification()
    attack = foolbox.attacks.FGSM(fmodel)

    for i in range(len(images)):
        print 'starting attack'
        image = images[i]
        label = int(np.argmax(labels[i]))

        adversarial = attack(image, label)
        if adversarial is None:
            print "fail attacking"
            continue
        # if the attack fails, adversarial will be None and a warning will be printed
        plt.figure()

        plt.subplot(1, 3, 1)
        plt.title('Original')
        plt.imshow(np.reshape(image,(28,28)), cmap = 'gray')  # division by 255 to convert [0, 255] to [0, 1]
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Adversarial')
        plt.imshow(np.reshape(adversarial,(28,28)), cmap = 'gray')  # ::-1 to convert BGR to RGB
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('Difference')
        difference = adversarial - image
        plt.imshow(np.reshape(difference,(28,28)), cmap = 'gray')
        plt.axis('off')

        plt.show()

        print 'ending attack'
        if i > 20:
            exit()



if __name__ == '__main__':
    main()