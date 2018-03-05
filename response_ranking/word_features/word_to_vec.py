'''
Word Embedding
--------------
    word embedding-based feature that calculates the average cosine distance between word embeddings
of all non-stopword pairs ⟨vSj, vQi⟩.
    vSj represent the word vector of jth word in S and vQj represent the word vector of ith word in Q.
'''
from src.static_variable import PATH_GOOGLE_WORD2VEC, STOPWORDS
import gensim

from textblob import Sentence

# load Google's pre-trained Word2Vec model
model = gensim.models.KeyedVectors.load_word2vec_format(PATH_GOOGLE_WORD2VEC,
                                                        binary=True)
def prep_text(txt):
    '''
    Process a text string to a list of non-stopwords, be a google's vocab
    :param txt: text String
    :return: list of non-stopwords and be a google's vocab
    '''
    sentence = Sentence(txt.lower()).words
    # loop for filter a non-stopwords and google's vocab
    return [word for word in sentence if (word not in STOPWORDS and word in model.vocab)]

def get_score(query, candidate):
    '''
    Calculates the average cosine distance between word embeddings
    :param query: String of query
    :param res: String of candidate
    :return:
    '''
    tmp_query = prep_text(query)
    tmp_candidate = prep_text(candidate)

    sum_cosine = 0.0
    for word1 in tmp_query:
        if word1 in model.vocab:
            for word2 in tmp_candidate:
                # compute a cosine similarity between 2 word
                sum_cosine += model.similarity(word1, word2)

    # return a average cosine score
    return sum_cosine / (len(query) * len(candidate))

# # # # # # # # # # # # # # # Unit Test # # # # # # # # # # # # # # # # # #
# given query(Q) and candidate(S)
# query = 'Do you know the history of Beijing?'
# candidate = 'Beijing is a historical city that can be traced back to 3,000 years ago.'
# get_score(query, candidate)