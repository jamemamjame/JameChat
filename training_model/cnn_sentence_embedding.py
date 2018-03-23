import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.learn import *

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_sentence_embedding(features, labels, mode, params):
    '''
    :param features: dict of sentence features with shape (batch_size, max_words, dim_of_word)
    features['seq1'] return batch of query sentence
    features['seq2'] return batch of positive response sentence
    features['seq3'] return batch of negative response sentence
    :param labels: nothing
    :param mode:
    :param params:
    :return:
    '''
    print('CURRENT MODE: %s' % mode.upper())

    M = params['M']  # a constant for computed with loss

    with tf.variable_scope("embedding"):
        # Define Filter or Kernel
        # shape = max_words * emb_dim * n_filter
        emb_w = tf.get_variable("emb_w", shape=[n_grams, emb_dim, n_chanel, n_filter],
                                initializer=tf.random_normal_initializer)
        emb_b = tf.get_variable("emb_b", shape=[n_filter],
                                initializer=tf.zeros_initializer)
        for v in tf.trainable_variables():
            tf.summary.histogram(name=v.name.replace(':0', ''), values=v)

    def embed_sentence(x):
        '''
        Embed sequence of word_vector
            conv -> max_pooling -> flatten -> *dropout -> dense(unit=embed_dim)

            *dropout only used in TRAIN mode
        :param x:
        :return:
        '''
        convol = tf.nn.conv2d(
            input=x,
            filter=emb_w,
            padding="SAME",
            strides=[1, 1, 1, 1],  # must use [1, ?, ?, 1]
        )
        convol = tf.nn.relu(convol + emb_b)

        # Pooling, output.shape = n_sample * max_word * 1 * n_filter
        pooling = tf.nn.max_pool(convol, ksize=[1, 1, emb_dim, 1], strides=[1, 1, 1, 1], padding="VALID")

        # Dense layer
        # reshape [n_samples, max_word, 1, n_filter] ->> [n_samples, max_word * 1 * n_filter]
        # pool_flat = tf.reshape(pooling, shape=[-1, max_word * 1 * n_filter])
        pool_flat = layers.flatten(pooling)
        dropout = tf.layers.dropout(inputs=pool_flat, rate=params['drop_out_rate'], training=(mode == ModeKeys.TRAIN))

        # how many dim of vector for represent this sentence (x)
        embed_dim = 50
        dense = tf.layers.dense(inputs=dropout, units=embed_dim, activation=tf.nn.relu)

        return dense

    def cosine_similarity(vec1, vec2):
        '''
        calculate cosine_similarity of each sample
        by A•B / (norm(A) * norm(B))
        :param vec1: batch of vector1
        :param vec2: batch of vector2
        :return:
        '''

        # calculate (norm(A) * norm(B))
        # output.shape = [n_sample, ]
        vec_norm = tf.norm(vec1, axis=1) * tf.norm(vec2, axis=1)

        # multiply sub_vec vs sub_vec.
        # output.shape = [n_sample , emb_dim]
        mul = tf.multiply(vec1, vec2)

        # sum values in emb_dim for each sample so output.shape = [n_sample, ]
        reduce_sum = tf.reduce_sum(mul, axis=1)

        # calculate cosine similarity.
        # output.shape = [n_sample, ]
        cosine_sim = reduce_sum / vec_norm

        return cosine_sim

    loss = None
    train_op = None

    # every mode must push seq1 be one of features dict
    seq1 = features['seq1']

    # Calculate Loss (for TRAIN, EVAL modes)
    if mode != ModeKeys.INFER:
        seq2 = features['seq1']  # get a pos_response
        seq3 = features['seq3']  # get a neg_response

        # get embedded vector: output.shape = [n_sample, emb_dim]
        vec1 = embed_sentence(seq1)  # query
        vec2 = embed_sentence(seq2)  # pos_response
        vec3 = embed_sentence(seq3)  # neg_response

        # calculate cosine similarity of each vec pairs, output.shape = [n_sample, ]
        cosine_sim_pos = cosine_similarity(vec1, vec2)  # need a large value
        cosine_sim_neg = cosine_similarity(vec1, vec3)  # need a tiny value

        # calculate loss of each pair pos_neg. output.shape = [n_sample, ]
        each_loss = -cosine_sim_pos + cosine_sim_neg  # << too small too good, boundary [-1, 1]

        # sum all loss. get output be scalar
        total_loss = tf.reduce_sum(each_loss)
        # tf.summary.scalar('total_loss', total_loss)

        # final loss
        loss = tf.maximum(0., M + total_loss)

    # Configure the Training Optimizer (for TRAIN modes)
    if mode == ModeKeys.TRAIN:
        # configuration the training Op
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            optimizer=tf.train.AdamOptimizer,
            learning_rate=params['learning_rate'],
            summaries=[
                'learning_rate',
                'loss',
                "gradients",
                "gradient_norm",
            ]
        )

    # Generate Predictions which is a embedding of given sentence
    predictions = {'emb_vec': embed_sentence(seq1)}

    # Generate a eval metric consist of loss
    # This metric will be constructed when we train, validate
    eval_metric_ops = {
        'loss': loss
    }

    # Return a ModelFnOps object
    return ModelFnOps(predictions=predictions, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops, mode=mode)


def get_sentence_embedder():
    '''
    Get a SKCompat it a class that can use .fit() for train || .score() for evaluate|| .predict() for predict
    :return: SKCompat model
    '''
    return SKCompat(Estimator(model_fn=cnn_sentence_embedding,
                              model_dir=PATH_CNN_SENTENCE_EMB,
                              config=RunConfig(save_checkpoints_secs=300, keep_checkpoint_max=3),
                              params=model_params,
                              feature_engineering_fn=None
                              ))


# path of cnn sentence embedding
PATH_CNN_SENTENCE_EMB = '/Users/jamemamjame/Computer-Sci/_chula course/SENIOR PROJECT/JameChat/training_model/_model/cnn_sentence_emb'

n_filter = 10  # n_filter is used to capture N phrase in any position in sentence
n_grams = 2  # n_grams is used to group N word be 1 phrase
max_word = 20
emb_dim = 150  # embedding size in word2vec
n_chanel = 1
n_train_sample = 100
n_test_sample = 20
training_batch_size = 5
test_batch_size = 2

model_params = dict(
    learning_rate=0.05,
    drop_out_rate=0.2,
    M=training_batch_size * 1 * 0.75
)


# ======================================================================
# # # # # # # # # # # # # # # # SIMULATION # # # # # # # # # # # # # # #


def get_simulation_data():
    '''
    Simulate that we have a real data training_dict, testing_dict and predict_dict
    :return:
    '''
    # define a fake dict of dataset
    training_dict = {
        'seq1': np.random.rand(n_train_sample, max_word, emb_dim, n_chanel).astype(np.float32),
        'seq2': np.random.rand(n_train_sample, max_word, emb_dim, n_chanel).astype(np.float32),
        'seq3': np.random.rand(n_train_sample, max_word, emb_dim, n_chanel).astype(np.float32)
    }
    testing_dict = {
        'seq1': np.random.rand(n_test_sample, max_word, emb_dim, n_chanel).astype(np.float32),
        'seq2': np.random.rand(n_test_sample, max_word, emb_dim, n_chanel).astype(np.float32),
        'seq3': np.random.rand(n_test_sample, max_word, emb_dim, n_chanel).astype(np.float32)
    }
    predict_dict = {
        'seq1': np.random.rand(1, max_word, emb_dim, n_chanel).astype(np.float32),
    }

    return training_dict, testing_dict, predict_dict


training_dict, testing_dict, predict_dict = get_simulation_data()

# ======================================================================
# # # # # # # # # # # # # # # # # TRAIN MODEL # # # # # # # # # # # # # # #
#
# validation test on testing data every_n_step or every save_checkpoints_secs
# validation_monitor = monitors.ValidationMonitor(x=testing_dict, y=None, every_n_steps=50, name='validation')
#
# # Training model is ran by step, not epoch
sentence_embedder = get_sentence_embedder()
#
# # Train the model with n_step step and do validation test
# sentence_embedder.fit(x=training_dict, y=None, batch_size=training_batch_size, steps=100, monitors=None)
#
# # Try to predict
# pred = sentence_embedder.predict(input_fn=predict_input_fn, as_iterable=False)
