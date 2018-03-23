'''
LSTM Context Embedding
----------------------------------------

ต้องการฝึกสอนโมเดลที่ใช้ฝังตัวบริบทให้กลายเป็นเวกเตอร์ และฝังประโยคคำตอบให้กลายเป็นเวกเตอร์ โดย 2 สิ่งนี้จะกลายเป็นเวกเตอร์ที่มีความสัมพันธ์กัน
โดยอาศัย CNN + LSTM ดังนี้
1) ใช้ CNN สกัดเอาลักษณะสำคัญที่ปรากฎประโยคต่างๆ ในบริบทและแปลงประโยคต่างๆ เหล่านั้นให้กลายเป็นเวกเตอร์
                context | list of sentence ➜ list of embedded_sentence
2) สำหรับแต่ละ embedded_sentence ซึ่งเป็นเวกเตอร์ เราจะนำมารวบให้กลายเป็นเวกเตอร์ใหม่เพียงตัวเดียว ผ่านการฝึกสอนโดยใช้ LSTM ในการเรียนรู้
โดยคาดหวังว่า LSTM จะสามารถจับความเป็นเรื่องราวของแต่ละประโยคและสามารถสรุปได้ว่าประโยคถัดไป (next sentence) ควรมีหน้าตาเป็นอย่างไร
'''
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers, rnn
from tensorflow.contrib.learn import *

tf.logging.set_verbosity(tf.logging.INFO)
tf.set_random_seed(4321)


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

    emb_size = 200
    n_lstm_units = 200  # number of hidden units
    M = params['M']  # a constant for computed with loss

    # LSTM cell for context
    with tf.variable_scope("lstm_context_emb_cell"):
        # create a LSTM cell
        cell_query = rnn.LSTMCell(num_units=n_lstm_units)
        projection_cell_query = rnn.OutputProjectionWrapper(cell=cell_query, output_size=emb_size,
                                                            activation=tf.nn.relu)

        for v in tf.trainable_variables():
            tf.summary.histogram(name=v.name.replace(':0', ''), values=v)

    # LSTM cell for response
    with tf.variable_scope("lstm_response_emb_cell"):
        # create a LSTM cell
        cell_resp = rnn.LSTMCell(num_units=n_lstm_units)
        projection_cell_resp = rnn.OutputProjectionWrapper(cell=cell_resp, output_size=emb_size,
                                                           activation=tf.nn.relu)

        for v in tf.trainable_variables():
            tf.summary.histogram(name=v.name.replace(':0', ''), values=v)

    def cnn_embed_sentence(x):
        # Convolutional Layer #1
        # the number of filter implies about the number of keyword that we attended
        conv1 = tf.layers.conv2d(
            inputs=x,
            filters=128,
            kernel_size=[n_grams, w2v_emb_size],
            padding='SAME',
            activation=tf.nn.relu)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Dense Layer
        pool_flat = layers.flatten(inputs=pool1)
        embed = tf.layers.dense(inputs=pool_flat, units=cnn_emb_dim, activation=tf.nn.relu)
        return embed

    def lstm_embed_context(x):
        # (outputs, final_state) is returned from tf.nn.dynamic_rnn()
        # outputs is an collection of all outputs in every step emitted which shape = (batch, time_step, n_output_size)
        # final_state = (c_state, h_state) final
        # but in this project, we care only outputs
        outputs, _ = tf.nn.dynamic_rnn(cell=projection_cell_query, inputs=x, time_major=False, dtype=tf.float32)

        # transpose (batch, time_step, n_output_size) -> (time_step, batch, n_output_size)
        #   ↳ unpack to list [(batch, outputs)..] * steps
        outputs = tf.transpose(outputs, [1, 0, 2])

        # get the last output from last time_step only.
        # shape = (batch, n_output_size)
        outputs = outputs[-1]

        # assume that this outputs is a emb_vector
        return outputs

    def embed_response(x):
        # (outputs, final_state) is returned from tf.nn.dynamic_rnn()
        # outputs is an collection of all outputs in every step emitted which shape = (batch, time_step, n_output_size)
        # final_state = (c_state, h_state) final
        # but in this project, we care only outputs
        outputs, _ = tf.nn.dynamic_rnn(cell=projection_cell_resp, inputs=x, time_major=False, dtype=tf.float32)

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

    # Calculate Loss (for TRAIN, EVAL modes)
    if mode != ModeKeys.INFER:
        contexts = features[CONTEXT_KEY]  # get a query
        seq2 = features[POS_RESP_KEY]  # get a pos_response
        seq3 = features[NEG_RESP_KEY]  # get a neg_response

        # convert every sentence (sequence of word2vec) in each pack (n_last_chat) to be a embed_sentence
        # convert from 1 sentence or 1 sequence_of_word2vec which shape = [n_word, w2v_emb_size] to [emb_dim,]
        cnn_contexts = []
        for sample_idx, sample_item in enumerate(contexts):
            # sample_item is a packed of last chat
            # we embed it into vector, so we got a data shape = [n_last_chat, emb_dim]
            cnn_contexts.append(cnn_embed_sentence(sample_item))

        # cnn_contexts become a tensor which shape = [n_sample, n_last_chat (or time_steps), emb_dim]
        cnn_contexts = tf.convert_to_tensor(cnn_contexts, dtype=tf.float32)

        # get LSTM embedded vector: output.shape = [n_sample , emb_dim]
        with tf.variable_scope("context_emb_vec"):
            vec1 = lstm_embed_context(cnn_contexts)  # query
        with tf.variable_scope("response_emb_vec"):
            vec2 = embed_response(seq2)  # pos_response
            vec3 = embed_response(seq3)  # neg_response

        # calculate cosine similarity of each vec pairs, output.shape = [n_sample, ]
        cosine_sim_pos = cosine_similarity(vec1, vec2)  # need a large value
        cosine_sim_neg = cosine_similarity(vec1, vec3)  # need a tiny value

        # calculate loss of each pair pos_neg. output.shape = [n_sample, ]
        each_loss = -cosine_sim_pos + cosine_sim_neg  # << too small too good, boundary [-1, 1]

        # sum all loss. get output be scalar
        total_loss = tf.reduce_sum(each_loss)

        # final loss
        loss = tf.maximum(0., M + total_loss)

    # Configure the Training Optimizer (only TRAIN modes)
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

    predictions = {}

    # for prediction mode
    # Generate Predictions which is a embedding of given sentence
    if mode == ModeKeys.INFER:
        # if user want to get a query_embedding
        if features.keys().__contains__(CONTEXT_KEY):
            predictions = {'emb_vec': lstm_embed_context(features[CONTEXT_KEY])}

        # if user want to get a response_embedding
        elif features.keys().__contains__(POS_RESP_KEY):
            predictions = {'emb_vec': embed_response(features[POS_RESP_KEY])}

    # Return a ModelFnOps object
    return ModelFnOps(predictions=predictions, loss=loss, train_op=train_op, eval_metric_ops=None, mode=mode)


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
PATH_LSTM_SENTENCE_EMB = './trained_model/lstm_context_emb'

n_filter = 10  # n_filter is used to capture N phrase in any position in sentence
n_grams = 2  # n_grams is used to group N word be 1 phrase

n_last_chat = 6
n_filter = 10  # n_filter is used to capture N phrase in any position in sentence
n_grams = 2  # n_grams is used to group N word be 1 phrase
n_chanel = 1
cnn_emb_dim = 150

# for context: we assume that the properly response is depended on the last 6 chat.
context_max_word = 15 * n_last_chat
context_time_steps = context_max_word

# for response
resp_max_word = 15
resp_time_steps = resp_max_word

w2v_emb_size = 150  # embedding size in word2vec
n_train_sample = 100
n_test_sample = 20
training_batch_size = 5
test_batch_size = 2

CONTEXT_KEY = 'context'
POS_RESP_KEY = 'pos_response'
NEG_RESP_KEY = 'neg_response'

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
        CONTEXT_KEY: np.random.rand(n_train_sample, n_last_chat, context_max_word, w2v_emb_size, n_chanel).astype(
            np.float32),
        POS_RESP_KEY: np.random.rand(n_train_sample, n_last_chat, resp_max_word, w2v_emb_size, n_chanel).astype(
            np.float32),
        NEG_RESP_KEY: np.random.rand(n_train_sample, n_last_chat, resp_max_word, w2v_emb_size, n_chanel).astype(
            np.float32),
    }
    testing_dict = {
        CONTEXT_KEY: np.random.rand(n_train_sample, context_max_word, w2v_emb_size).astype(np.float32),
        POS_RESP_KEY: np.random.rand(n_train_sample, resp_max_word, w2v_emb_size).astype(np.float32),
        NEG_RESP_KEY: np.random.rand(n_train_sample, resp_max_word, w2v_emb_size).astype(np.float32),
    }
    predict_dict = {
        CONTEXT_KEY: np.random.rand(1, context_max_word, w2v_emb_size).astype(np.float32),
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
