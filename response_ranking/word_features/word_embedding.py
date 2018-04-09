'''
Word Embedding
--------------
    word embedding-based feature that calculates the average cosine distance between word embeddings
of all non-stopword pairs ⟨vSj, vQi⟩.
    vSj represent the word vector of jth word in S and vQj represent the word vector of ith word in Q.
'''
from src.static_variable import load_stopwords, load_word_embedding
from textblob import Sentence

# get set of english stopwords
STOPWORDS = load_stopwords()

# load pre-trained Word Embedding model
WORD_EMB = load_word_embedding(load_glove=True)


def prep_text(txt):
    '''
    Process a text string to a list of non-stopwords, be a google's vocab
    :param txt: text String
    :return: list of non-stopwords and be a google's vocab
    '''
    # loop for filter a non-stopwords and google's vocab
    return [word for word in Sentence(txt.lower()).words if word not in STOPWORDS]


def get_score(query, candidate):
    '''
    Calculates the average cosine distance between word embeddings
    :param query: String of query
    :param res: String of candidate
    :return: Float of average cosine score
    '''
    query = prep_text(query)
    candidate = prep_text(candidate)

    count = 0
    sum_cosine = 0.0
    for word1 in query:
        if word1 in WORD_EMB.vocab:
            for word2 in candidate:
                if word2 in WORD_EMB.vocab:
                    # compute a cosine similarity between 2 word
                    sum_cosine += WORD_EMB.similarity(word1, word2)
                    count += 1

    # return a average cosine score
    return 0 if count == 0 else sum_cosine / count

# # # # # # # # # # # # # # # Unit Test # # # # # # # # # # # # # # # # # #
# given query(Q) and candidate(S)
# query = 'Do you know the history of Beijing?'
# candidate = 'Beijing is a historical city that can be traced back to 3,000 years ago.'
# get_score(query, candidate)
