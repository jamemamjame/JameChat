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
from tensorflow.contrib import rnn
from tensorflow.contrib.learn import *
import os

tf.logging.set_verbosity(tf.logging.INFO)
tf.set_random_seed(4321)


def lstm_context_embedding(features, labels, mode, params):
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

    # create a LSTM cell for context
    with tf.variable_scope("emb_cell1_context"):
        cell1 = rnn.LSTMCell(num_units=n_lstm_units, activation=tf.nn.tanh)
        if mode == ModeKeys.TRAIN:
            cell1 = rnn.DropoutWrapper(cell=cell1, input_keep_prob=input_keep_prob,
                                       output_keep_prob=output_keep_prob)

    def lstm_embed_context(x):
        outputs, _ = tf.nn.dynamic_rnn(cell=cell1, inputs=x, time_major=False, dtype=tf.float32)
        outputs = tf.transpose(outputs, [1, 0, 2])
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

    # Calculate Loss (for TRAIN, EVAL modes)
    if mode != ModeKeys.INFER:
        seq1 = features[CONTEXT_KEY]   # get context
        seq2 = features[POS_RESP_KEY]  # get a pos_response
        seq3 = features[NEG_RESP_KEY]  # get a neg_response

        # get embedded vector: output.shape = [n_sample , emb_dim]
        vec1 = lstm_embed_context(seq1)
        vec2 = lstm_embed_context(seq2)
        vec3 = lstm_embed_context(seq3)

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
        if features.keys().__contains__(CONTEXT_KEY):
            seq1 = features[CONTEXT_KEY]
            predictions = {'emb_vec': lstm_embed_context(seq1)}
        elif features.keys().__contains__(POS_RESP_KEY):
            seq2 = features[POS_RESP_KEY]
            predictions = {'emb_vec': lstm_embed_context(seq2)}

    # Return a ModelFnOps object
    return ModelFnOps(predictions=predictions, loss=loss, train_op=train_op, eval_metric_ops=None, mode=mode)

def get_sentence_embedder():
    '''
    Get a SKCompat it a class that can use .fit() for train || .score() for evaluate|| .predict() for predict
    :return: SKCompat model
    '''
    return SKCompat(Estimator(model_fn=lstm_context_embedding,
                              model_dir=PATH_LSTM_SENTENCE_EMB,
                              config=RunConfig(save_checkpoints_secs=300, keep_checkpoint_max=2),
                              params=model_params,
                              feature_engineering_fn=None))

# path of LSTM sentence embedding
PATH_LSTM_SENTENCE_EMB = 'trained_model/context_model2'

w2v_emb_size = 300  # embedding size in word2vec
model_params = dict(
    learning_rate=0.001,
    input_keep_prob=0.85,
    output_keep_prob=0.85,
    M=1.0
)

# # get estimator
sentence_embedder = get_sentence_embedder()

CONTEXT_KEY = 'context'
POS_RESP_KEY = 'pos_response'
NEG_RESP_KEY = 'neg_response'