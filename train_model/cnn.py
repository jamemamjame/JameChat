import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn import *

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode, params):
    '''

    :param features: sentence features with shape (batch_size, max_words, dim_of_word)
    :param labels: nothing
    :param mode:
    :param params:
    :return:
    '''
