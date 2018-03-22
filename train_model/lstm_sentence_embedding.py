import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers, rnn
from tensorflow.contrib.learn import *

tf.logging.set_verbosity(tf.logging.INFO)


def lstm_sentence_embedding(features, labels, mode, params):
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

    emb_size = 128
    n_hidden_units = 128
    M = params['M']  # a constant

    with tf.variable_scope("embedding"):
        # create a LSTM cell
        cell = rnn.LSTMCell(num_units=n_hidden_units)
        projection_cell = rnn.OutputProjectionWrapper(cell=cell, output_size=emb_size, activation=tf.nn.relu)

        # # set initial state
        # # LSTM cell is divided into 2 parts in 1 tuple: (c_state, h_state)
        # init_state = projection_cell.zero_state(batch_size=training_batch_size, dtype=tf.float32)

        for v in tf.trainable_variables():
            tf.summary.histogram(name=v.name.replace(':0', ''), values=v)

    def embed_sentence(x):
        # (outputs, final_state) is returned from tf.nn.dynamic_rnn()
        # outputs is an collection of all outputs in every step emitted which shape = (batch, time_step, n_output_size)
        # final_state = (c_state, h_state) final
        # but in this project, we care only outputs
        outputs, _ = tf.nn.dynamic_rnn(cell=projection_cell, inputs=x, time_major=False, dtype=tf.float32)

        # transpose (batch, time_step, n_output_size) -> (time_step, batch, n_output_size)
        #   ↳ unpack to list [(batch, outputs)..] * steps
        outputs = tf.transpose(outputs, [1, 0, 2])

        # get the last output from last time_step only.
        # shape = (batch, n_output_size)
        outputs = outputs[-1]

        # assume that this outputs is a vector
        return outputs

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

        # get embedded vector: output.shape = [n_sample , emb_dim]
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
    return SKCompat(Estimator(model_fn=lstm_sentence_embedding,
                              model_dir=PATH_LSTM_SENTENCE_EMB,
                              config=RunConfig(save_checkpoints_secs=300, keep_checkpoint_max=3),
                              params=model_params,
                              feature_engineering_fn=None
                              ))


# path of LSTM sentence embedding
PATH_LSTM_SENTENCE_EMB = '/Users/jamemamjame/Computer-Sci/_chula course/SENIOR PROJECT/JameChat/train_model/_model/lstm_sentence_emb'

n_filter = 10  # n_filter is used to capture N phrase in any position in sentence
n_grams = 2  # n_grams is used to group N word be 1 phrase
max_word = 20
time_steps = max_word
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
        'seq1': np.random.rand(n_train_sample, time_steps, emb_dim).astype(np.float32),
        'seq2': np.random.rand(n_train_sample, time_steps, emb_dim).astype(np.float32),
        'seq3': np.random.rand(n_train_sample, time_steps, emb_dim).astype(np.float32),
    }
    testing_dict = {
        'seq1': np.random.rand(n_train_sample, time_steps, emb_dim).astype(np.float32),
        'seq2': np.random.rand(n_train_sample, time_steps, emb_dim).astype(np.float32),
        'seq3': np.random.rand(n_train_sample, time_steps, emb_dim).astype(np.float32),
    }
    predict_dict = {
        'seq1': np.random.rand(1, time_steps, emb_dim).astype(np.float32),
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
sentence_embedder.fit(x=training_dict, y=None, batch_size=training_batch_size, steps=2, monitors=None)
#
# # Try to predict
# pred = sentence_embedder.predict(x=predict_dict, batch_size=1)
