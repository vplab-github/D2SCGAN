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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


d = keras.models.load_model("Models/discriminator.h5")
Y_tr = np.load('testImages.npy')
y_true = np.load('testLabels.npy')
[rubbish, y_pred] = d.predict(Y_tr)
print(np.equal(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1)).mean())