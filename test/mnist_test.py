'''
Test cyclegan using MNIST
'''
import os
from PIL import Image
import numpy as np
import keras.backend as k
from keras.layers import Input
from keras.datasets import mnist
from keras.models import Model
from keras.optimizers import Adam

import context
from cyclegan.models import mnist_discriminator, mnist_generator
from cyclegan.losses import *

def load_mnist():
    '''
    returns mnist_data
    '''
    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if k.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    return input_shape, np.concatenate((x_train, x_test), axis=0)

def load_input_images(from_jpegs=False):
    current_dir = os.path.dirname(__file__)
    image_names = os.listdir(os.path.join(current_dir, 'images'))
    images = np.ndarray((len(image_names), 28, 28), dtype=np.uint8)
    for i, filename in enumerate(image_names):
        images[i] = Image.open(os.path.join(current_dir,
                                            'images',
                                            filename)).resize((28, 28)).convert('L')
    images = images.astype('float32')
    images /= 255
    if k.image_data_format() == 'channels_first':
        images = images.reshape(len(image_names), 1, 28, 28)
    else:
        images = images.reshape(len(image_names), 28, 28, 1)
    return images

def prettify(image):
    image *= 255
    image = np.uint8(image)
    image = np.reshape(image, (28, 28))
    return image

def test_cyclegan():
    mnist_shape, mnist_images = load_mnist()
    faces = load_input_images()

    nb_epochs = 100
    batch_size = 1024
    adam_lr = 0.0002
    adam_beta_1 = 0.5
    history_size = int(batch_size * 7/3)
    generation_history_mnist = None
    generation_history_faces = None

    mnist_input = Input(mnist_shape)
    face_input = Input(mnist_shape)

    generator_faces = mnist_generator(mnist_shape)
    discriminator_faces = mnist_discriminator(mnist_shape)
    generator_mnist = mnist_generator(mnist_shape)
    discriminator_mnist = mnist_discriminator(mnist_shape)

    discriminator_faces.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1), loss=discriminator_loss)
    discriminator_mnist.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1), loss=discriminator_loss)
    generator_faces.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1), loss='mean_squared_error')
    generator_mnist.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1), loss='mean_squared_error')

    fake_face = generator_faces(mnist_input)
    fake_mnist = generator_mnist(face_input)

    # Only train discriminator during first phase
    # (only affects new models when they are compiled)
    discriminator_faces.trainable = False
    discriminator_mnist.trainable = False

    face_gen_trainer = Model(mnist_input, discriminator_faces(fake_face))
    mnist_gen_trainer = Model(face_input, discriminator_mnist(fake_mnist))
    mnist_cyc = Model(mnist_input, generator_mnist(fake_face))
    faces_cyc = Model(face_input, generator_faces(fake_mnist))

    face_gen_trainer.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1), loss='mean_squared_error')
    mnist_gen_trainer.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1), loss='mean_squared_error')
    mnist_cyc.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1), loss=cycle_loss)
    faces_cyc.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1), loss=cycle_loss)

    # training time

    mnist_discrim_loss = []
    faces_discrim_loss = []
    mnist_gen_loss = []
    faces_gen_loss = []
    mnist_cyc_loss = []
    faces_cyc_loss = []


    for epoch in range(nb_epochs):
        print("\n\n================================================")
        print("Epoch", epoch, '\n')
        if epoch == 0: # Initialize history for training the discriminator
                mnist_indices = np.random.choice(mnist_images.shape[0], history_size)
                generation_history_mnist = mnist_images[mnist_indices]
                faces_indices = np.random.choice(faces.shape[0], history_size)
                generation_history_faces = faces[faces_indices]

        for batch in range(int(mnist_images.shape[0]/batch_size)):
            print("\nBatch", batch)
            # Get batch.
            mnist_indices = np.random.choice(mnist_images.shape[0], batch_size)
            mnist_batch_real = mnist_images[mnist_indices]
            faces_indices = np.random.choice(faces.shape[0], batch_size)
            faces_batch_real = faces[faces_indices]

            # Update history with new generated images.
            mnist_batch_gen = generator_mnist.predict_on_batch(faces_batch_real)
            faces_batch_gen = generator_faces.predict_on_batch(mnist_batch_real)
            generation_history_mnist = np.concatenate((generation_history_mnist[batch_size:], mnist_batch_gen))
            generation_history_faces = np.concatenate((generation_history_faces[batch_size:], faces_batch_gen))

            # Train discriminators.
            mnist_discrim_loss.append(discriminator_mnist.train_on_batch(np.concatenate((generation_history_mnist[:batch_size], mnist_batch_real)),
                                               np.concatenate((np.zeros(batch_size), np.ones(batch_size)))))
            print("MNIST Discriminator Loss:", mnist_discrim_loss[-1])
            faces_discrim_loss.append(discriminator_faces.train_on_batch(np.concatenate((generation_history_faces[:batch_size], faces_batch_real)),
                                               np.concatenate((np.zeros(batch_size), (np.ones(batch_size))))))
            print("Faces Discriminator Loss:", faces_discrim_loss[-1])

            # Train generators.
            mnist_gen_loss.append(mnist_gen_trainer.train_on_batch(faces_batch_real, np.ones(batch_size)))
            print("MNIST Generator Loss:", mnist_gen_loss[-1])
            faces_gen_loss.append(face_gen_trainer.train_on_batch(mnist_batch_real, np.ones(batch_size)))
            print("Faces Generator Loss:", faces_gen_loss[-1])

            cyc_multiplier = 10
            mnist_cyc_loss.append(mnist_cyc.train_on_batch(mnist_batch_real, mnist_batch_real, sample_weight=np.array([cyc_multiplier]*batch_size)))
            print("MNIST Cyclic Loss:", mnist_cyc_loss[-1])
            faces_cyc_loss.append(faces_cyc.train_on_batch(faces_batch_real, faces_batch_real, sample_weight=np.array([cyc_multiplier]*batch_size)))
            print("Faces Cyclic Loss:", faces_cyc_loss[-1])

        choice = np.random.choice(mnist_images.shape[0])
        mnist_in = mnist_images[choice].reshape((1, 28, 28, 1))
        face_out = generator_faces.predict(mnist_in)
        mnist_in = Image.fromarray(prettify(mnist_in), mode='L')
        face_out = Image.fromarray(prettify(face_out), mode='L')
        mnist_in.save(str(epoch)+'_mnist_in.jpg')
        face_out.save(str(epoch)+'_face_out.jpg')

    generator_faces.save('generator_faces.h5')

if __name__ == "__main__":
    test_cyclegan()
    print('hello :D')
