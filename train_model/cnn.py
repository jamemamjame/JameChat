import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.learn import *
from tensorflow.python.estimator.inputs.numpy_io import numpy_input_fn

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_embedding(features, labels, mode, params):
    '''
    :param features: sentence features with shape (batch_size, max_words, dim_of_word)
    :param labels: nothing
    :param mode:
    :param params:
    :return:
    '''
    print('CURRENT MODE: %s' % mode.upper())

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
        convol = tf.nn.conv2d(
            input=x,
            filter=emb_w,
            padding="SAME",
            strides=[1, 1, 1, 1],
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
        by reduce_sum(from A•B) / (norm(A) * norm(B))
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
    predictions = None

    # every mode must push query be one of features dict
    query = features['query']

    # Calculate Loss (for TRAIN, EVAL modes)
    if mode != ModeKeys.INFER:
        pos_response = features['pos_response']
        neg_response = features['neg_response']

        # get embedded vector: output.shape = [n_sample , emb_dim]
        vec1 = embed_sentence(query)  # query
        vec2 = embed_sentence(pos_response)  # pos_response
        vec3 = embed_sentence(neg_response)  # neg_response

        # calculate cosine similarity of each vec pairs, output.shape = [n_sample, ]
        cosine_sim_pos = cosine_similarity(vec1, vec2)  # need a large value
        cosine_sim_neg = cosine_similarity(vec1, vec3)  # need a tiny value

        # calculate loss of each pair pos_neg. output.shape = [n_sample, ]
        each_loss = -cosine_sim_pos + cosine_sim_neg  # << too small too good, boundary [-1, 1]

        # sum all loss. get output be scalar
        total_loss = tf.reduce_sum(each_loss)
        # tf.summary.scalar('total_loss', total_loss)

        # final loss
        M = params['M']
        # loss = tf.maximum(0., M + total_loss)
        loss = total_loss

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
    predictions = {'emb_vec': embed_sentence(query)}

    # Generate a eval metric consist of loss
    # This metric will be constructed when we train, validate
    eval_metric_ops = {
        'loss': loss
    }

    # Return a ModelFnOps object
    return ModelFnOps(predictions=predictions, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops, mode=mode)


n_filter = 10  # n_filter is used to capture N phrase in any position in sentence
n_grams = 2  # n_grams is used to group N word be 1 phrase
max_word = 20
emb_dim = 10
n_chanel = 1

# ======================================================================
# # # # # # # # # # # # # # # # SIMULATION # # # # # # # # # # # # # # #

n_train_sample = 100
n_test_sample = 20
training_batch_size = 5
test_batch_size = 2

# define a fake dict of dataset
training_dict = {
    'query': np.random.rand(n_train_sample, max_word, emb_dim, n_chanel).astype(np.float32),
    'pos_response': np.random.rand(n_train_sample, max_word, emb_dim, n_chanel).astype(np.float32),
    'neg_response': np.random.rand(n_train_sample, max_word, emb_dim, n_chanel).astype(np.float32)
}
testing_dict = {
    'query': np.random.rand(n_test_sample, max_word, emb_dim, n_chanel).astype(np.float32),
    'pos_response': np.random.rand(n_test_sample, max_word, emb_dim, n_chanel).astype(np.float32),
    'neg_response': np.random.rand(n_test_sample, max_word, emb_dim, n_chanel).astype(np.float32)
}
predict_dict = {
    'query': np.random.rand(1, max_word, emb_dim, n_chanel).astype(np.float32),
}

# Train by step, not epoch
train_input_fn = numpy_input_fn(x=training_dict, y=np.zeros(shape=[n_train_sample, 1], dtype=np.float32),
                                batch_size=training_batch_size, shuffle=True, num_epochs=None)
test_input_fn = numpy_input_fn(x=testing_dict, y=np.zeros(shape=[n_test_sample, 1], dtype=np.float32),
                               batch_size=test_batch_size, shuffle=True, num_epochs=None)
predict_input_fn = numpy_input_fn(x=predict_dict, shuffle=False, num_epochs=None, batch_size=1)

# validation test on testing data every_n_step or every save_checkpoints_secs
validation_monitor = monitors.ValidationMonitor(input_fn=test_input_fn, every_n_steps=50, name='validation')
# ======================================================================
model_params = dict(
    learning_rate=0.05,
    drop_out_rate=0.2,
    M=training_batch_size * 1 * 0.75
)
# Training model is ran by step, not epoch
sentence_embedder = Estimator(model_fn=cnn_embedding,
                              model_dir='./train_model/_model/sentence_emb',
                              config=RunConfig(save_checkpoints_secs=4),
                              params=model_params,
                              feature_engineering_fn=None)

# Train the model with n_step step and do validation test
sentence_embedder.fit(input_fn=train_input_fn, steps=2000, monitors=[validation_monitor])
pred = sentence_embedder.predict(input_fn=predict_input_fn, as_iterable=False)
