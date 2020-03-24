import keras
from keras.models import Model
from keras.layers import Conv2D, Input, Deconvolution2D, merge
from keras.optimizers import SGD, adam
import prepare_data as pd
import numpy
import math
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, GaussianDropout, GaussianNoise, Add
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, Deconv2D, Conv2DTranspose, AveragePooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD, adam, adadelta
from keras.datasets import mnist
from keras.models import Model
import numpy as np
import numpy.linalg as LA
import numpy.linalg as LA
from PIL import Image
import argparse
import os
import os.path
import math
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy as misc
from keras.callbacks import TensorBoard
import random
# import Image
import math
from scipy import misc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.99
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=5, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma)  # window shape [size, size]
    # print window
    K1 = 0.01
    K2 = 0.03
    L = 255  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                              (sigma1_sq + sigma2_sq + C2)),
                 (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                             (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool(img1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.stack(mssim, axis=0)
    mcs = tf.stack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level - 1] ** weight[0:level - 1]) *
             (mssim[level - 1] ** weight[level - 1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def multires_patchwise_mse(y_true, y_pred):
    ch1_t, ch2_t, ch3_t = tf.split(y_true, [1, 1, 1], axis=3)      # Separating channels from the image tensor "y_true"
    ch1_p, ch2_p, ch3_p = tf.split(y_pred, [1, 1, 1], axis=3)      # Separating channels from the image tensor "y_pred"
    final_loss_1, final_loss_2, final_loss_3 = 0.0, 0.0, 0.0
    alpha = 0.0
    for kernel_size in [(input_dim/4), (input_dim/2), input_dim]:
        kernel = [1, kernel_size, kernel_size, 1]
        if kernel_size == (input_dim/4):
            strides = [1, 10, 10, 1]
            alpha = 0.00001
        if kernel_size == (input_dim/2):
            strides = [1, 20, 20, 1]
            alpha = 0.0001
        if kernel_size == input_dim:
            strides = [1, 10, 10, 1]
            alpha = 0.001
        patches_true_1 = tf.extract_image_patches(ch1_t, kernel, strides, [1, 1, 1, 1], padding='SAME')
        patches_true_2 = tf.extract_image_patches(ch2_t, kernel, strides, [1, 1, 1, 1], padding='SAME')
        patches_true_3 = tf.extract_image_patches(ch3_t, kernel, strides, [1, 1, 1, 1], padding='SAME')
        patches_pred_1 = tf.extract_image_patches(ch1_p, kernel, strides, [1, 1, 1, 1], padding='SAME')
        patches_pred_2 = tf.extract_image_patches(ch2_p, kernel, strides, [1, 1, 1, 1], padding='SAME')
        patches_pred_3 = tf.extract_image_patches(ch3_p, kernel, strides, [1, 1, 1, 1], padding='SAME')
        loss_1 = 0.0
        loss_2 = 0.0
        loss_3 = 0.0
        patches = math.ceil((kernel_output - kernel_size) / strides[1])
        if patches > 0:
            for i in range(int(patches)):
                for j in range(int(patches)):
                    loss_1 = loss_1 + (0.2989 * mse(patches_true_1[0, i, j], patches_pred_1[0, i, j]) * alpha)
                    loss_2 = loss_2 + (0.5870 * mse(patches_true_2[0, i, j], patches_pred_2[0, i, j]) * alpha)
                    loss_3 = loss_3 + (0.1141 * mse(patches_true_3[0, i, j], patches_pred_3[0, i, j]) * alpha)
            loss_1 = loss_1 / (patches * patches)
            loss_2 = loss_2 / (patches * patches)
            loss_3 = loss_3 / (patches * patches)
        else:
            loss_1 = 0.2989 * mse(patches_true_1, patches_pred_1) * alpha
            loss_2 = 0.5870 * mse(patches_true_2, patches_pred_2) * alpha
            loss_3 = 0.1141 * mse(patches_true_3, patches_pred_3) * alpha
        final_loss_1 = final_loss_1 + loss_1
        final_loss_2 = final_loss_2 + loss_2
        final_loss_3 = final_loss_3 + loss_3
    total_loss = 0.000001 * (final_loss_1 + final_loss_2 + final_loss_3)
    return total_loss


def kld(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)


def jsd(y_true, y_pred):
    img1 = tf.image.rgb_to_grayscale(y_true)
    img2 = tf.image.rgb_to_grayscale(y_pred)
    val_range = [0.0, 255.0]
    hist1 = tf.cast(tf.histogram_fixed_width(img1, val_range, nbins=64), tf.float32)
    hist2 = tf.cast(tf.histogram_fixed_width(img2, val_range, nbins=64), tf.float32)
    m = tf.cast(tf.add(hist1, hist2), tf.float32)
    m_mod = tf.divide(m, tf.constant(2.))
    t1 = kld(hist1, m_mod)
    t2 = kld(hist2, m_mod)
    jsd = tf.div(tf.add(t1, t2), tf.constant(2.))
    jsd = tf.where(tf.is_nan(jsd), 0., jsd)
    return 0.05 * jsd


def chi_squared_distance(y_true, y_pred):
    val_range = [-0.5, 0.5]
    hist1 = tf.cast(tf.histogram_fixed_width(y_true, val_range, nbins=16), tf.float32)
    hist2 = tf.cast(tf.histogram_fixed_width(y_pred, val_range, nbins=16), tf.float32)
    hist1_normalize = K.l2_normalize(hist1)
    hist2_normalize = K.l2_normalize(hist2)
    elmw_sqd = tf.cast(tf.squared_difference(hist1_normalize, hist2_normalize), tf.float32)
    elmw_div = tf.cast(tf.truediv(elmw_sqd, hist1_normalize), tf.float32)
    loss = tf.cast(tf.reduce_sum(elmw_div), tf.float32) * 0.5
    loss = tf.where(tf.is_nan(loss), 0., loss)
    loss = tf.where(tf.is_inf(loss), 0., loss)
    return 0.001 * (loss / 16)


def structural_loss(y_true, y_pred):
    img1 = tf.image.rgb_to_grayscale(y_true)
    img2 = tf.image.rgb_to_grayscale(y_pred)
    ms_ssim_loss = 1 - tf_ms_ssim(img1, img2) # heuristics, gives better generation result
    mse_loss = keras.losses.mse(y_true, y_pred)
    mr_pmse_loss = multires_patchwise_mse(y_true, y_pred)
    chi_squared_loss = chi_squared_distance(img1, img2)
    jsd_loss = jsd(y_true, y_pred)
    st_loss = K.abs(ms_ssim_loss + 0.1 * ms_pmse_loss + mse_loss + 0.1 * chi_squared_loss + 0.0001 * jsd_loss)
    return st_loss


def generator(input_col, input_row):
    _input = Input(shape=(input_col, input_row, 3), name='input')

    # Channel 1: Shallow channel

    x = Conv2D(1, (1, 1), input_shape=(input_row, input_row, 3), padding='same')(_input)
    x = Flatten()(x)
    x = Dense(2048)(x)
    x = Activation('tanh')(x)
    x = Dense(128 * (dim_output/4) * (dim_output/4))(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)
    x = Reshape(((dim_output/4), (dim_output/4), 128), input_shape=(128 * (dim_output/4) * (dim_output/4),))(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (5, 5), padding='same')(x)
    x = Activation('tanh')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (5, 5), padding='same')(x)
    out_1 = Conv2D(nb_filter=3, nb_row=5, nb_col=5, activation='tanh', border_mode='same')(x)
    model1 = Model(input=_input, output=out_1)
    g_optim = adam(lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=1e-5)
    model1.compile(loss=structural_loss, optimizer=g_optim)
    plot_model(model1, to_file='channel1.png')

    # Channel 2: Deep channel

    x = Conv2D(3, (1, 1), input_shape=(input_row, input_row, 3), padding='same')(_input)
    x = Activation('tanh')(x)
    x = Flatten()(x)
    x = Dense(2048)(x)
    x = Activation('tanh')(x)
    x = Dense(128 * (dim_output/4) * (dim_output/4))(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)
    x = Reshape(((dim_output/4), (dim_output/4), 128), input_shape=(128 * (dim_output/4) * (dim_output/4),))(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, (5, 5), padding='same')(x)
    x = Activation('tanh')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, (5, 5), padding='same')(x)
    x = Activation('tanh')(x)
    x = Conv2D(64, (5, 5), padding='same')(x)
    x = Activation('tanh')(x)
    x = Conv2D(32, (5, 5), padding='same')(x)
    x = Activation('tanh')(x)
    out_2 = Conv2D(nb_filter=3, nb_row=5, nb_col=5, activation='tanh', border_mode='same')(x)
    model2 = Model(input=_input, output=out_2)

    g_optim = adam(lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=1e-5)
    model2.compile(loss=structural_loss, optimizer=g_optim)
    plot_model(model2, to_file='channel2.png')

    EES = model1(_input)
    EED = model2(_input)

    add_layer_output = keras.layers.average(inputs=[EED, EES])
    model = Model(input=_input, output=add_layer_output)
    Adam = adam(lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=1e-5)
    model.compile(optimizer=Adam, loss=structural_loss)
    #
    # plot_model(model, to_file='generator.png')

    return model


def discriminator_model(input_row, input_col):
    _input = Input(shape=(input_col, input_row, 3), name='input')
    x = Conv2D(64, (5, 5), padding='same', input_shape=(input_row, input_col, 3))(_input)
    x = Activation('tanh')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (5, 5))(x)
    x = Activation('tanh')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(1024)(x)
    out = Activation('tanh')(x)
    x = Dense(1)(out)
    x = Activation('sigmoid')(x)
    y = Dense(51)(out)
    y = Activation('sigmoid')(y)
    model = Model(input=_input, output=[x, y])
    # plot_model(model, to_file='model2.png')
    return model


def generator_containing_discriminator(g, d):
    d.trainable = False
    model = Model(inputs=g.inputs, outputs=d(g.outputs))
    return model


def train(BATCH_SIZE, input_row, input_col):

    # Trainers
    # adadelta = keras.optimizers.adadelta(lr=0.0001, decay=1e-5)
    # Adam = adam(lr=0.0001, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=1e-5)
    # d_optim = SGD(lr=0.0001, momentum=0.9, nesterov=True)

    d = discriminator_model(dim_output, dim_output)
    g = generator(input_row, input_col)
    Adam = adam(lr=0.0001, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=1e-5)
    g.compile(optimizer=Adam, loss=structural_loss)
    print(g.summary())
    # d.load_weights('/home/vplab/PycharmProjects/DSC-GAN/Models_CAD/D1/discriminator100')
    # g.load_weights('/home/vplab/PycharmProjects/DSC-GAN/Models_CAD/G1/generator100')
    Y_tr = np.load('trImages.npy') ## Load gallery data
    X_tr = np.load('tsImages.npy') ## Load probe data
    X_lb = np.load('trLabels.npy') ## Load gallery labels
    Y_lb = np.load('tsLabels.npy') ## Load probe labels
    result = []
    X_trn = []
    Y_trn = []
    X_lbl = []
    Y_lbl = []

    for x in range(0, len(X_tr)):
        num = random.randint(0, len(X_tr) - 1)
        while num in result:
            num = random.randint(0, len(X_tr) - 1)
        result.append(num)

    for i in result:
        X_im = X_tr[i]
        X_label = X_lb[i]
        X_trn.append(X_im)
        X_lbl.append(X_label)
        Y_im = Y_tr[i]
        Y_label = Y_lb[i]
        Y_trn.append(Y_im)
        Y_lbl.append(Y_label)
    X_train = np.asarray(X_trn)
    Y_train = np.asarray(Y_trn)
    X_labels = np.asarray(X_lbl)
    Y_labels = np.asarray(Y_lbl)

    # print(Y_train.shape[0])

    g_loss = 0.
    d_loss = 0.00001
    alpha_loss = 0
    s_x = []

    g_x = []
    d_x = []

    d_on_g = generator_containing_discriminator(g, d)
    d_optim = SGD(lr=0.00001, momentum=0.9, nesterov=True)
    g_optim = adam(lr=0.00001, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=1e-5)

    d_on_g.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=g_optim)
    d.trainable = True
    d.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=d_optim)

    for epoch in range(num_epochs):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))

        for index in range(int(X_train.shape[0] / BATCH_SIZE)):
            noise = np.asarray(Y_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE])
            labels_batch = np.asarray(Y_labels[index * BATCH_SIZE:(index + 1) * BATCH_SIZE])
            image_batch = np.asarray(X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE])
            generated_images = g.predict(noise, verbose=0)

            if index % 30 == 0:
                image = combine_images(generated_images)
                image_or = combine_images(image_batch)
                image = image * 127.5 + 127.5
                image_or = image_or * 127.5 + 127.5
                Image.fromarray(image.astype(np.uint8)).save("imgs/" +
                                                             str(epoch) + "_" + str(index) + ".png")
                Image.fromarray(image_or.astype(np.uint8)).save("imgs/" +
                                                                str(epoch) + "_" + str(index) + "_or.png")
            if epoch > 6:
                g_trend = sum(g_x[epoch-6:epoch])/5.
                d_trend = sum(d_x[epoch-6:epoch])/5.
                diff = g_trend - d_trend
                ratio = LA.norm(g_loss)/LA.norm(d_loss)

                if diff > 1 or ratio > 5:
                    d.trainable = False
                    # print('visited')
                if diff < -1 or ratio < 0.01:
                    g.trainable = False

            X = np.array(image_batch)
            y = np.array([1] * BATCH_SIZE)
            d.train_on_batch(X, [y, labels_batch])

            X = generated_images
            y = np.array([0] * BATCH_SIZE)
            d_loss = d.train_on_batch(X, [y, labels_batch])

            y = np.array([1] * BATCH_SIZE)

            d.trainable = False
            alpha_loss = g.train_on_batch(noise, image_batch)
            g_loss = d_on_g.train_on_batch(noise, [y, labels_batch])

            d.trainable = True
            g.trainable = True

            print("epoch %d batch %d d_loss : %f g_loss : %f s_loss : %f" % (epoch, index, LA.norm(d_loss), LA.norm(g_loss), alpha_loss))

            if epoch % 200 == 99:
                g.save('Models_CAD/G/generator_cad5_%d.h5' % epoch, True)
                d.save('Models_CAD/D/discriminator_cad5_%d.h5' % epoch, True)
                np.save('Models_CAD/generator_cad5_.npy', g_x, True)
                np.save('Models_CAD/discriminator_cad5_.npy', d_x, True)
                np.save('Models_CAD/structural_cad5_.npy', s_x, True)

        g_x.append(LA.norm(g_loss))
        d_x.append(LA.norm(d_loss))
        s_x.append(alpha_loss)


def l1_norm(weight_matrix):
    return K.sum(K.abs(weight_matrix))


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:4]
    image = np.zeros((height * shape[0], width * shape[1], shape[2]),
                     dtype=generated_images.dtype)
    for index, rgb in enumerate(generated_images):
        img = rgb
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], 0] = img[:, :, 0]
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], 1] = img[:, :, 1]
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], 2] = img[:, :, 2]
    return image


## Model parameters

kernel_output = 140 
dim_output = 140
input_dim = 35
train(10, 35, 35)
num_epochs = 100000
