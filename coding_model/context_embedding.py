'''
LSTM Context Embedding
----------------------------------------

ต้องการฝึกสอนโมเดลที่ใช้ฝังตัวบริบทให้กลายเป็นเวกเตอร์ และฝังประโยคคำตอบให้กลายเป็นเวกเตอร์ โดย 2 สิ่งนี้จะกลายเป็นเวกเตอร์ที่มีความสัมพันธ์กัน
โดยอาศัย CNN + LSTM ดังนี้
1) ใช้ CNN สกัดเอาลักษณะสำคัญที่ปรากฎประโยคต่างๆ ในบริบทและแปลงประโยคต่างๆ เหล่านั้นให้กลายเป็นเวกเตอร์
                context | list of sentence ➜ list of embedded_sentence
2) สำหรับแต่ละ embedded_sentence ซึ่งเป็นเวกเตอร์ เราจะนำมารวบให้กลายเป็นเวกเตอร์ใหม่เพียงตัวเดียว ผ่านการฝึกสอนโดยใช้ LSTM ในการเรียนรู้
โดยคาดหวังว่า LSTM จะสามารถจับความเป็นเรื่องราวของแต่ละประโยคและสามารถสรุปได้ว่าประโยคถัดไป (next sentence) ควรมีหน้าตาเป็นอย่างไร


ยัง predict CONTEXT ไม่ได้ เพราะ input ตอนนี้เป็น seq of w2v จริงๆต้องใช้ seq of cnn_emb
'''
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers, rnn, learn
from tensorflow.contrib.learn import *
import os

tf.logging.set_verbosity(tf.logging.INFO)
tf.set_random_seed(4321)


def cnn_lstm_context_embedding(features, labels, mode, params):
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

    n_lstm_units = 100  # number of hidden units
    M = params['M']  # a constant for computed with loss
    cnn_drop_out_rate = params['cnn_drop_out_rate']
    rnn_input_keep_prob = params['rnn_input_keep_prob']
    rnn_output_keep_prob = params['rnn_output_keep_prob']
    learning_rate = params['learning_rate']

    # LSTM cell for context embedding
    with tf.variable_scope("lstm_context_emb_cell"):
        # create a LSTM cell
        cell = rnn.LSTMCell(num_units=n_lstm_units)

        if mode == ModeKeys.TRAIN:
            cell = rnn.DropoutWrapper(cell, rnn_input_keep_prob, rnn_output_keep_prob)

        projection_cell = rnn.OutputProjectionWrapper(cell=cell, output_size=lstm_emb_size,
                                                      activation=tf.nn.sigmoid)
    #         for v in tf.trainable_variables():
    #             tf.summary.histogram(name=v.name.replace(':0', ''), values=v)

    # CNN filter variable for sentence embedding
    with tf.variable_scope("cnn_sentence_emb"):
        # create a filter/kernel
        cnn_emb_w = tf.get_variable("emb_w", shape=[n_grams, w2v_emb_size, n_chanel, n_filter],
                                    initializer=tf.random_normal_initializer)

        cnn_emb_b = tf.get_variable("emb_b", shape=[n_filter],
                                    initializer=tf.zeros_initializer)
        for v in tf.trainable_variables():
            tf.summary.histogram(name=v.name.replace(':0', ''), values=v)

    def cnn_embed_sentence(x):
        # make sure that shape of x have n_chanel at the last rank
        x = tf.reshape(x, [-1, max_word, w2v_emb_size, n_chanel])

        # Convolutional Layer #1
        # the number of filter implies about the number of keyword that we attended
        conv1 = tf.nn.conv2d(
            input=x,
            filter=cnn_emb_w,
            padding="VALID",
            strides=[1, 1, 1, 1],  # must use [1, ?, ?, 1]
        )
        conv1 = tf.nn.relu(conv1 + cnn_emb_b)

        # Pooling, output.shape = n_sample * max_word * 1 * n_filter
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 1, 1], strides=[1, 1, 1, 1], padding="VALID")

        # Dense Layer
        pool_flat = layers.flatten(inputs=pool1)
        dense1 = tf.layers.dense(inputs=pool_flat, units=256, activation=tf.nn.relu)
        dropout1 = tf.layers.dropout(inputs=dense1, rate=cnn_drop_out_rate, training=mode == learn.ModeKeys.TRAIN)

        embed = tf.layers.dense(inputs=dropout1, units=cnn_emb_dim, activation=tf.nn.relu)
        return embed

    def lstm_embed_context(x):
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

        # assume that this outputs is a emb_vector
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

    # Calculate Loss (for TRAIN, EVAL modes)
    if mode != ModeKeys.INFER:
        contexts = features[CONTEXT_KEY]  # get a context [n_sample, n_last_chat, max_word, w2v_emb_size]
        seq2 = features[POS_RESP_KEY]  # get a pos_response [n_sample, max_word, w2v_emb_size]
        seq3 = features[NEG_RESP_KEY]  # get a neg_response [n_sample, max_word, w2v_emb_size]

        # we want to convert every sentence (sequence of word2vec) in each pack (1 pack have n_last_chat sequence)
        # to be a embed_sentence
        # convert from 1 sentence or 1 sequence_of_word2vec which shape = [n_word, w2v_emb_size] to [emb_dim,]

        # reshape by combine each sentence in each sample together
        # [n_sample, n_last_chat, max_word, w2v_emb_size] -> [-1, max_word, w2v_emb_size]
        contexts = tf.reshape(contexts, [-1, max_word, w2v_emb_size])

        # use CNN for embed every sentence
        cnn_contexts = cnn_embed_sentence(contexts)

        # reshape to original shape = [n_sample, n_last_chat (or time_steps), cnn_emb_dim]
        cnn_contexts = tf.reshape(cnn_contexts, [-1, n_last_chat, cnn_emb_dim])

        # this is a batch of vector (context embedding) which shape = [n_sample, lstm_emb_size]
        vec1 = lstm_embed_context(cnn_contexts)  # context
        # this is a batch of vector (sentence embedding or response embedding) which shape = [n_sample, cnn_emb_size]
        vec2 = cnn_embed_sentence(seq2)  # pos_response
        vec3 = cnn_embed_sentence(seq3)  # neg_response

        # calculate cosine similarity of each vec pairs, output.shape = [n_sample,]
        cosine_sim_pos = cosine_similarity(vec1, vec2)  # need a large value
        cosine_sim_neg = cosine_similarity(vec1, vec3)  # need a tiny value

        # calculate loss of each pair pos_neg. output.shape = [n_sample,]
        losses = tf.maximum(0., M - cosine_sim_pos + cosine_sim_neg)  # << too small too good

        # final_loss = sum all loss. and get output be scalar
        loss = tf.reduce_mean(losses)

    # Configure the Training Optimizer (only TRAIN modes)
    if mode == ModeKeys.TRAIN:
        # configuration the training Op
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            optimizer=tf.train.AdamOptimizer,
            learning_rate=learning_rate,
            #             clip_gradients=1.0, # protect a exploding gradients problem
            summaries=[
                'learning_rate',
                'loss',
                "gradients",
                "gradient_norm",
            ]
        )

    # default predictions is a empty, but will generate if and only if mode == prediction
    predictions = {}

    # for prediction mode
    # Generate Predictions which is a embedding of given sentence
    if mode == ModeKeys.INFER:
        # if user want to get a query_embedding
        if features.keys().__contains__(CONTEXT_KEY):
            # reshape data in a true format
            contexts = features[CONTEXT_KEY]

            # reshape by combine each sentence in each sample together
            # [n_sample, n_last_chat, max_word, w2v_emb_size] -> [-1, max_word, w2v_emb_size]
            contexts = tf.reshape(contexts, [-1, max_word, w2v_emb_size, n_chanel])

            # use CNN for embed every sentence
            cnn_contexts = cnn_embed_sentence(contexts)

            # reshape to proper shape = [n_sample, n_last_chat (or time_steps), cnn_emb_dim]
            cnn_contexts = tf.reshape(cnn_contexts, [-1, n_last_chat, cnn_emb_dim])

            # call function for get a vector
            predictions = {'emb_vec': lstm_embed_context(cnn_contexts)}

        # if user want to get a response_embedding
        elif features.keys().__contains__(POS_RESP_KEY):
            predictions = {'emb_vec': cnn_embed_sentence(features[POS_RESP_KEY])}

    # Return a ModelFnOps object
    return ModelFnOps(predictions=predictions, loss=loss, train_op=train_op, eval_metric_ops=None, mode=mode)


def get_sentence_embedder():
    '''
    Get a SKCompat it a class that can use .fit() for train || .score() for evaluate|| .predict() for predict
    :return: SKCompat model
    '''
    return SKCompat(Estimator(model_fn=cnn_lstm_context_embedding,
                              model_dir=PATH_LSTM_SENTENCE_EMB,
                              config=RunConfig(save_checkpoints_secs=300, keep_checkpoint_max=2),
                              params=model_params,
                              feature_engineering_fn=None
                              ))


# path of LSTM sentence embedding
PATH_LSTM_SENTENCE_EMB = 'trained_model/context_model'

n_last_chat = 3
n_filter = 512  # n_filter is used to capture N phrase in any position in sentence
n_grams = 3  # n_grams is used to group N word be 1 phrase
n_chanel = 1
cnn_emb_dim = 100

# for context: we assume that the properly response is depended on the last 6 chat.
max_word = 11
context_time_steps = max_word
lstm_emb_size = cnn_emb_dim

# for response
resp_time_steps = max_word

w2v_emb_size = 300  # embedding size in word2vec

model_params = dict(
    learning_rate=10e-4,
    cnn_drop_out_rate=.2,
    rnn_input_keep_prob=.85,
    rnn_output_keep_prob=.85,
    M=1.0
)


# ======================================================================
# # # # # # # # # # # # # # # # SIMULATION # # # # # # # # # # # # # # #


CONTEXT_KEY = 'context'
POS_RESP_KEY = 'pos_response'
NEG_RESP_KEY = 'neg_response'

path_context = 'train_context.npy'
path_pos_resp = 'train_pos_resp.npy'

# training_dict = {
#     CONTEXT_KEY: np.load(path_context),
#     POS_RESP_KEY: np.load(path_pos_resp),
#     NEG_RESP_KEY: np.load(path_pos_resp),
# }
