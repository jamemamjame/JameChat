from train_model.cnn import get_sentence_embedder
from tensorflow.python.estimator.inputs.numpy_io import numpy_input_fn
from train_model.word_seq_perp import generate_dialogs_embedding
from scipy.spatial.distance import cosine

SENTENCE_EMB = get_sentence_embedder()


def get_score(query, candidate):
    # convert sentence to dict with key = 'seq1'
    query_dict = {'seq1': generate_dialogs_embedding([query])}
    candidate_dict = {'seq1': generate_dialogs_embedding([candidate])}

    # create input function
    query_input_fn = numpy_input_fn(x=query_dict, shuffle=False, num_epochs=None, batch_size=1)
    candidate_input_fn = numpy_input_fn(x=candidate_dict, shuffle=False, num_epochs=None, batch_size=1)

    # get a query sentence embedding
    query_emb = SENTENCE_EMB.predict(input_fn=query_input_fn, as_iterable=False)
    query_emb = query_emb['emb_vec'][0]

    # get a candidate sentence embedding
    candidate_emb = SENTENCE_EMB.predict(input_fn=candidate_input_fn, as_iterable=False)
    candidate_emb = candidate_emb['emb_vec'][0]

    return 1 - cosine(query_emb, candidate_emb)

# # # # # # # # # # # # # # # Unit Test # # # # # # # # # # # # # # # # # #
# given query(Q) and candidate(S)
# query = 'Do you know the history of Beijing?'
# candidate = 'Beijing is a historical city that can be traced back to 3,000 years ago.'
# get_score(query, candidate)
