'''
Word Matching
-------------
    word matching feature that counts the number (weighted by the IDF value of each word in S)
    of non-stopwords shared by S and Q.

    * Document here is list of candidates

- Note:
    how to calculate TF-IDF: https://en.wikipedia.org/wiki/Tf–idf
'''
from textblob import Sentence, Word
from src.static_variable import load_stopwords
from numpy import log

STOPWORDS = load_stopwords()

# define a dict of part of speech (POS) mapping
NOUN, VERB, ADJ, ADV = 'n', 'v', 'j', 'r'
POSMAP = {'N': NOUN, 'V': VERB, 'j': ADJ, 'r': ADV}

def prep_text(txt):
    '''
    Process a text string to list of word
    :param txt: text String
    :return: list of word that is non-stopwords and lemmatized
    '''
    word_list = []
    postags = Sentence(txt.lower()).pos_tags
    for word, pos in postags:
        if word in STOPWORDS:
            continue

        if pos[0] in POSMAP.keys():
            word_list.append(Word(word).lemmatize(POSMAP[pos[0]]))
        else:
            word_list.append(word)
    return Sentence(' '.join(word_list))

def tf(word, sentence):
    return sentence.words.count(word, case_sensitive=False) / len(sentence.words)


def idf(word, document):
    return log(1 + (len(document) / n_containing(word, document)))


def n_containing(word, document):
    return sum(1 for sentence in document if word in sentence.words)


def tfidf(word, sentence, document):
    '''
    :param word: String
    :param sentence: Sentence/ TextBlob
    :param document: TextBlob
    :return: float score
    '''
    return tf(word, sentence) * idf(word, document)


def get_score(query, candidate, document):
    '''
    :param query:
    :param candidate: String of sentence
    :param candidate_list: list of sentence String
    :return:
    '''

    query = prep_text(query)
    candidate = prep_text(candidate)
    document = [prep_text(doc) for doc in document]

    score = 0.0
    # loop for each word without stopwords
    for word1 in query.words:
        if word1 not in STOPWORDS:
            for word2 in candidate.words:
                if word2 not in STOPWORDS:
                    if word1 == word2:
                        score += tfidf(word1, candidate, document)

    return score

# # # # # # # # # # # # # # # Unit Test # # # # # # # # # # # # # # # # # #
# given query(Q), candidate(S) and candidates(D)
# query = 'Do you know the history of Beijing?'
# candidate = 'Beijing is a historical city that can be traced back to 3,000 years ago.'
# document = [
#     "Beijing is a historical city that can be traced back to 3,000 years ago.",
#     "The city's history dates back three millennia. As the last of the Four Great Ancient Capitals of China",
#     "Beijing has been the political center of the country for much of the past eight centuries",
#     "With mountains surrounding the inland city on three sides"
# ]
# get_score(query, candidate, document)
