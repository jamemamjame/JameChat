from coding_model.context_embedding import get_sentence_embedder
from coding_model.preprocess.word_seq_perp import generate_dialogs_embedding
from scipy.spatial.distance import cosine
import numpy as np

SENTENCE_EMB = get_sentence_embedder()


def get_score(context, candidate):
    # convert context to dict with key = 'context'
    context_dict = {'context': generate_dialogs_embedding(context)}
    context_dict['context'] = np.array([context_dict['context']]).astype(np.float32)

    candidate_dict = {'pos_response': generate_dialogs_embedding([candidate])}

    # get a query sentence embedding
    context_emb = SENTENCE_EMB.predict(context_dict)
    context_emb = context_emb['emb_vec'][0]

    # get a candidate sentence embedding
    candidate_emb = SENTENCE_EMB.predict(candidate_dict)
    candidate_emb = candidate_emb['emb_vec'][0]

    # return a cosine similarity from 1 - cosine_distance
    return 1 - cosine(context_emb, candidate_emb)

# # # # # # # # # # # # # # # Unit Test # # # # # # # # # # # # # # # # # #
# given query(Q) and candidate(S)
# candidate = 'I know the history of Beijing'
# context = [
#     "Beijing is a historical city that can be traced back to 3,000 years ago.",
#     "The city's history dates back three millennia. As the last of the Four Great Ancient Capitals of China",
#     "Beijing has been the political center of the country for much of the past eight centuries",
# ]
# get_score(context, candidate)