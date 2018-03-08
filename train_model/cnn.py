import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn, layers
from tensorflow.contrib.learn import *

tf.logging.set_verbosity(tf.logging.INFO)

n_sample = 4
n_filter = 10   # n_filter is used to capture N phrase in any position in sentence
n_grams = 2     # n_grams is used to group N word be 1 phrase
max_word = 20
emb_dim = 100
n_chanel = 1


def cnn_model_fn(features, labels, mode, params):
    '''

    :param features: sentence features with shape (batch_size, max_words, dim_of_word)
    :param labels: nothing
    :param mode:
    :param params:
    :return:
    '''
    query = features['query']
    pos_response = features['pos_response']
    neg_response = features['neg_response']

    with tf.variable_scope("embedding"):
        # this is filter ot kernel
        # shape = [max_words, emb_dim, n_filter=10]
        emb_w = tf.get_variable("emb_w", shape=[params['n_maxword'], params['emb_dim'], n_filter],
                                initializer=tf.random_normal_initializer)
        emb_b = tf.get_variable("emb_b", shape=[n_filter],
                                initializer=tf.zeros_initializer)

    def embed_sentence(x, W, B):
        convol = tf.nn.conv2d(
            input=x,
            filter=W,
            padding="SAME",
            strides=[1, 1, 1, 1],
        )
        convol = tf.nn.relu(convol + B)

        # pooling output size: n_sample x max_word x 1 x n_filter
        pooling = tf.nn.max_pool(convol, ksize=[1, 1, emb_dim, 1], strides=[1, 1, 1, 1], padding="VALID")

        # Dense layer
        # reshape [n_samples, max_word, 1, n_filter] ->> [n_samples, max_word * 1 * n_filter]
        # flatten = tf.reshape(pooling, shape=[-1, max_word * 1 * n_filter])
        flatten = layers.flatten(pooling)

        # how many dim of vector for represent this sentence (x)
        embed_dim = 50
        dense = tf.layers.dense(inputs=flatten, units=embed_dim, activation=tf.nn.relu)
        return dense

    def cosine_similarity(vec1, vec2):
        '''
        calculate cosine_similarity of each sample
        by reduce_sum(from A•B) / (norm(A) * norm(B))
        :param vec1: batch of vector1
        :param vec2: batch of vector2
        :return:
        '''

        # calculate (norm(A) * norm(B))
        ### output.shape = [n_sample, ]
        vec_norm = tf.norm(vec1, axis=1) * tf.norm(vec2, axis=1)

        # multiply sub_vec vs sub_vec.
        # output.shape = [n_sample , emb_dim]
        mul = tf.multiply(vec1, vec2)

        ### sum values in emb_dim for each sample so output.shape = [n_sample, ]
        reduce_sum = tf.reduce_sum(mul, axis=1)

        # calculate cosine similarity.
        # output.shape = [n_sample, ]
        cosine_sim = reduce_sum / vec_norm

        return cosine_sim

    # embedding: get vec.shape = [n_sample , emb_dim]
    vec1 = embed_sentence(query, emb_w, emb_b)  # query
    vec2 = embed_sentence(pos_response, emb_w, emb_b)  # pos_response
    vec3 = embed_sentence(neg_response, emb_w, emb_b)  # neg_response

    # calculate cosine similarity of each vec pairs, output.shape = [n_sample, ]
    cosine_sim_pos = cosine_similarity(vec1, vec2)  # need a large value
    cosine_sim_neg = cosine_similarity(vec1, vec3)  # need a tiny value

    # calculate loss of each pair pos_neg. output.shape = [n_sample, ]
    each_loss = -cosine_sim_pos + cosine_sim_neg  # << too small too good, boundary [-1, 1]

    # sum all loss. get output be scalar
    total_loss = tf.reduce_sum(each_loss)

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
