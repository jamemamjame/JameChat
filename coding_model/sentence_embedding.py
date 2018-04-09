'''
LSTM Sentence Embedding
----------------------------------------

ต้องการฝึกสอนโมเดลที่ใช้ฝังประโยค ประโยคหนึ่งให้กลายเป็นเวกเตอร์ตัวหนึ่ง โดยประโยค 2 ประโยคที่เป็นคำถามและคำตอบของกันและกัน
เมื่อถูกฝังแล้วจะต้องกลายเป็นเวกเตอร์ที่หาค่า cosine_similarity ได้สูงๆ
โดยการฝึกสอนจะใช้ LSTM ตัวเดียวกัน (weight ตัวเดียวกัน) ในการฝึกสอน ไม่ได้แยกกันมาจากคนละโมเดล
'''
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
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

    M = params['M']  # a constant for computed with loss
    input_keep_prob = params['input_keep_prob']
    output_keep_prob = params['output_keep_prob']
    n_lstm_units = 100  # number of hidden units

    # create a LSTM cell (only 1 cell but train both query, pos_response, neg_response)
    with tf.variable_scope("emb_cell"):
        cell = rnn.LSTMCell(num_units=n_lstm_units, activation=tf.nn.softmax)
        if mode == ModeKeys.TRAIN:
            cell = rnn.DropoutWrapper(cell=cell, input_keep_prob=input_keep_prob,
                                      output_keep_prob=output_keep_prob)
        projection_cell = rnn.OutputProjectionWrapper(cell=cell, output_size=lstm_emb_size, activation=tf.nn.softmax)

#         for v in tf.trainable_variables():
#             tf.summary.histogram(name=v.name.replace(':0', ''), values=v)

    def lstm_embed_sentence(x):
        # (outputs, final_state) is returned from tf.nn.dynamic_rnn()
        #     |         └→ final_state = (c_state, h_state) final
        #     └→ outputs is an collection of all outputs in every step emitted
        #        which shape = (batch, time_step, n_output_size)
        # but in this project, we care only outputs
        outputs, _ = tf.nn.dynamic_rnn(cell=projection_cell, inputs=x, time_major=False, dtype=tf.float32)

        # transpose (batch, time_step, n_output_size) -> (time_step, batch, n_output_size)
        #   └→ unpack to list [(batch, outputs)..] * steps
        outputs = tf.transpose(outputs, [1, 0, 2])

        # get the last output from last time_step only.
        # shape = (batch, n_output_size)
        outputs = outputs[-1]

        # assume that this outputs is a embed_vector
        return outputs

    def cosine_similarity(vec1, vec2):
        '''
        Calculate cosine_similarity of each sample
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
    seq1 = features[QUERY_KEY]

    # Calculate Loss (for TRAIN, EVAL modes)
    if mode != ModeKeys.INFER:
        seq2 = features[POS_RESP_KEY]  # get a pos_response
        seq3 = features[NEG_RESP_KEY]  # get a neg_response

        # get embedded vector: output.shape = [n_sample , emb_dim]
        vec1 = lstm_embed_sentence(seq1)  # query
        vec2 = lstm_embed_sentence(seq2)  # pos_response
        vec3 = lstm_embed_sentence(seq3)  # neg_response

        # calculate cosine similarity of each vec pairs, output.shape = [n_sample, ]
        cosine_sim_pos = cosine_similarity(vec1, vec2)  # need a large value
        cosine_sim_neg = cosine_similarity(vec1, vec3)  # need a tiny value

        # LOSS
        # calculate loss of each pair pos_neg. output.shape = [n_sample,]
        losses = tf.maximum(0., M - cosine_sim_pos + cosine_sim_neg)  # << too small too good

        # final_loss = sum all loss. and get output be scalar
        loss = tf.reduce_mean(losses)

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
    predictions = {}
    if mode == ModeKeys.INFER:
        predictions = {'emb_vec': lstm_embed_sentence(seq1)}

    # Return a ModelFnOps object
    return ModelFnOps(predictions=predictions, loss=loss, train_op=train_op, eval_metric_ops=None, mode=mode)

def get_sentence_embedder():
    '''
    Get a SKCompat it a class that can use .fit() for train || .score() for evaluate|| .predict() for predict
    :return: SKCompat model
    '''
    return SKCompat(Estimator(model_fn=lstm_sentence_embedding,
                              model_dir=PATH_LSTM_SENTENCE_EMB,
                              config=RunConfig(save_checkpoints_secs=300, keep_checkpoint_max=2),
                              params=model_params,
                              feature_engineering_fn=None))

# path of LSTM sentence embedding
PATH_LSTM_SENTENCE_EMB = 'trained_model/sentence_model'

max_word = 11
time_steps = max_word
w2v_emb_dim = 300  # embedding size in word2vec
lstm_emb_size = 100

model_params = dict(
    learning_rate=0.001,
    input_keep_prob=0.85,
    output_keep_prob=0.85,
    M=1.0
)

QUERY_KEY = 'seq1'
POS_RESP_KEY = 'seq2'
NEG_RESP_KEY = 'seq3'

path_query = 'train_query.npy'
path_pos_resp = 'train_resp.npy'

# training_dict = {
#     QUERY_KEY: np.load(path_query),
#     POS_RESP_KEY: np.load(path_pos_resp),
#     NEG_RESP_KEY: np.load(path_pos_resp), # load pos_response but in future will shuffle
# }


# ======================================================================
# # # # # # # # # # # # # # # # SIMULATION # # # # # # # # # # # # # # #


# def get_simulation_data():
#     '''
#     Simulate that we have a real data training_dict, testing_dict and predict_dict
#     :return:
#     '''
#     # define a fake dict of dataset
#     training_dict = {
#         QUERY_KEY: np.random.rand(n_train_sample, time_steps, w2v_emb_dim).astype(np.float32),
#         POS_RESP_KEY: np.random.rand(n_train_sample, time_steps, w2v_emb_dim).astype(np.float32),
#         NEG_RESP_KEY: np.random.rand(n_train_sample, time_steps, w2v_emb_dim).astype(np.float32),
#     }
#     testing_dict = {
#         QUERY_KEY: np.random.rand(n_train_sample, time_steps, w2v_emb_dim).astype(np.float32),
#         POS_RESP_KEY: np.random.rand(n_train_sample, time_steps, w2v_emb_dim).astype(np.float32),
#         NEG_RESP_KEY: np.random.rand(n_train_sample, time_steps, w2v_emb_dim).astype(np.float32),
#     }
#     predict_dict = {
#         QUERY_KEY: np.random.rand(1, time_steps, w2v_emb_dim).astype(np.float32),
#     }
#
#     return training_dict, testing_dict, predict_dict
#
#
# training_dict, testing_dict, predict_dict = get_simulation_data()

# ======================================================================
# # # # # # # # # # # # # # # # # TRAIN MODEL # # # # # # # # # # # # # # #
#
# validation test on testing data every_n_step or every save_checkpoints_secs
# validation_monitor = monitors.ValidationMonitor(x=testing_dict, y=None, every_n_steps=50, name='validation')
#
# # Training model is ran by step, not epoch
# sentence_embedder = get_sentence_embedder()
#
# # Train the model with n_step step and do validation test
# sentence_embedder.fit(x=training_dict, y=None, batch_size=training_batch_size, steps=2, monitors=None)
#
# # # Evaluate model
# sentence_embedder.score(x=testing_dict, y=None, batch_size=None)
# #
# # # Try to predict
# pred = sentence_embedder.predict(x=predict_dict, batch_size=1)
