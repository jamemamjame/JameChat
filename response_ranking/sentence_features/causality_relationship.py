from training_model.cnn_sentence_embedding import get_sentence_embedder
from training_model.word_seq_perp import generate_dialogs_embedding
from scipy.spatial.distance import cosine

SENTENCE_EMB = get_sentence_embedder()


def get_score(query, candidate):
    # convert sentence to dict with key = 'seq1'
    query_dict = {'seq1': generate_dialogs_embedding([query])}
    candidate_dict = {'seq1': generate_dialogs_embedding([candidate])}

    # get a query sentence embedding
    query_emb = SENTENCE_EMB.predict(query_dict)
    query_emb = query_emb['emb_vec'][0]

    # get a candidate sentence embedding
    candidate_emb = SENTENCE_EMB.predict(candidate_dict)
    candidate_emb = candidate_emb['emb_vec'][0]

    # return a cosine similarity from 1 - cosine_distance
    return 1 - cosine(query_emb, candidate_emb)

# # # # # # # # # # # # # # # Unit Test # # # # # # # # # # # # # # # # # #
# given query(Q) and candidate(S)
query = 'Do you know the history of Beijing?'
candidate = 'Beijing is a historical city that can be traced back to 3,000 years ago.'
# get_score(query, candidate)