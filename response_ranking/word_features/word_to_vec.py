'''
Word Embedding
--------------
    word embedding-based feature that calculates the average cosine distance between word embeddings
of all non-stopword pairs ⟨vSj, vQi⟩.
    vSj represent the word vector of jth word in S and vQj represent the word vector of ith word in Q.
'''
from src import filepath as path
import gensim
from sentence_prepocessing import Sentence_prepocess

# path of google word2vec
google_w2v_path = path.GOOGLE_WORD_TO_VEC

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format(google_w2v_path,
                                                        binary=True)
sentence_proc = Sentence_prepocess(keep_stopword=False)


def get_score(query, candidate):
    '''
    Calculates the average cosine distance between word embeddings
    :param query: String of query
    :param res: String of candidate
    :return:
    '''
    query = sentence_proc.cleaning(query)
    candidate = sentence_proc.cleaning(candidate)

    sum_cosine = 0.0
    for word1 in query:
        for word2 in candidate:
            # compute a cosine similarity between 2 word
            sum_cosine += model.similarity(word1, word2)
    avg_cosine = sum_cosine / (len(query) * len(candidate))
    return avg_cosine


# # given query(Q) and candidate(S)
# query = 'Do you know the history of Beijing?'
# candidate = 'Beijing is a historical city that can be traced back to 3,000 years ago.'
#
# w2v = word2vec()
