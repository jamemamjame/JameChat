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
    query = features['query']
    response = features['response']

    # maybe we need to reshape to add n_chanel at the last shape (like a grey-scale image)
    # query = tf.reshape(query, (query.shape[0], query.shape[1], query.shape[2], 1))

    n_filter = 10  # n_filter is used to capture N pharse in any position in sentence
    n_grams = 2  # n_grams is used to group N word be 1 phrase

    with tf.variable_scope("embedding"):
        # shape = (maxwords, emb_dim, n_filter=10)
        emb_w = tf.get_variable("emb_w", shape=[params['n_maxword'], params['emb_dim'], n_filter],
                                initializer=tf.random_normal_initializer)
        emb_b = tf.get_variable("emb_b", shape=[params['n_maxword'], params['emb_dim'], n_filter],
                                initializer=tf.zeros_initializer)

    # Convolution Layer (try to embed query with the same weight on response)
    conv_query = tf.layers.conv2d(
        inputs=query,
        filters=emb_w,
        kernel_size=[1, params['emb_dim']],
        padding="same",
        strides=[0, 1],
        activation=tf.nn.relu,
    )
    # Pooling Layer #1
    # pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    pool_query = tf.layers.max_pooling2d(inputs=conv_query, pool_size=[1, params['emb_dim']])

    # Convolution Layer (try to embed response with the same weight on query)
    conv_response = tf.layers.conv2d(
        inputs=query,
        filters=emb_w,
        kernel_size=[3, 3],
        padding="same",
        strides=[0, 1],
        activation=tf.nn.relu,
    )

    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        pass

    loss, train_op = None, None
    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            optimizer=tf.train.AdamOptimizer,
            learning_rate=params['learning_rate'],
            summaries=[
                "learning_rate",
                "loss",
                "gradients",
                "gradient_norm",
            ])
