'''
Ranking
-------
สำหรับวิจัย 1

Rank(S,Q)= ∑[ λk · hk(S,Q) ]
'''

# Importing the library
import response_ranking.word_features.word_embedding as W2V
import response_ranking.word_features.word_matching as WM
import response_ranking.sentence_features.sentence_emb as ST_EMB
import response_ranking.context_features.context_emb as CT_EMB
from random import shuffle

from textblob import Word, Sentence
from src.static_variable import load_stopwords, load_word_embedding
from coding_model.preprocess.word_seq_perp import generate_dialogs_embedding

# get set of english stopwords
STOPWORDS = load_stopwords()

# load pre-trained Word Embedding model
WORD_EMB = load_word_embedding(load_glove=True)

# =========================== Word Matching ===========================
# define a dict of part of speech (POS) mapping
NOUN, VERB, ADJ, ADV = 'n', 'v', 'j', 'r'
POSMAP = {'N': NOUN, 'V': VERB, 'j': ADJ, 'r': ADV}


def WM_prep_text(txt):
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


# =========================== Word 2 Vec ===========================
def W2V_prep_text(txt):
    '''
    Process a text string to a list of non-stopwords, be a google's vocab
    :param txt: text String
    :return: list of non-stopwords and be a google's vocab
    '''
    # loop for filter a non-stopwords and google's vocab
    return [word for word in Sentence(txt.lower()).words if word not in STOPWORDS]


# =========================== Sentence Embedding ===========================
#
#
# =========================== Context Embedding ===========================
#
#

def Rank(query, candidates, context):
    '''
    Rank the sentence in candidates set from high->low score
    :param query: String of query
    :param candidates: list of sentence that we need to rank for get a best response given query
    :return: ranked list of candidate
    '''

    # declare a list of (candidate, score)
    results = []

    # for Word-Matching
    WM_query = WM_prep_text(query)
    WM_document = [WM_prep_text(doc) for doc in candidates]

    # for Word2Vec
    W2V_query = W2V_prep_text(query)

    # for sentence embedding
    ST_query_dict = {'seq1': generate_dialogs_embedding([query], maxword=11)[0]}

    # for context embedding
    context = ' '.join(context)
    CT_context_dict = {'context': generate_dialogs_embedding([context], maxword=40)[0]}

    # loop for calculate score of each candidate
    for candidate in candidates:
        WM_candidate = WM_prep_text(candidate)
        W2V_candidate = W2V_prep_text(candidate)

        candidate_array = generate_dialogs_embedding([candidate], maxword=11)[0]
        ST_candidate_dict = {'seq1': candidate_array}
        CT_candidate_dict = {'pos_response': candidate_array}

        wm_score = WM.get_score(query=WM_query, candidate=WM_candidate, document=WM_document)
        w2v_score = W2V.get_score(query=W2V_query, candidate=W2V_candidate, WORD_EMB=WORD_EMB)
        st_score = ST_EMB.get_score(query_dict=ST_query_dict, candidate_dict=ST_candidate_dict)
        ct_score = CT_EMB.get_score(context_dict=CT_context_dict, candidate_dict=CT_candidate_dict)

        score = wm_score + w2v_score + st_score + ct_score

        results.append((candidate, score))

    # sorted a scores list by key = score
    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results


# # # # # # # # # # # # # # # Unit Test # # # # # # # # # # # # # # # # # #
import os, pickle
import numpy as np

MAIN_DATASET_PATH = '/Users/jamemamjame/Computer-Sci/_chula course/SENIOR PROJECT/Dataset/prep_dialog'

# loaf a log with (50000,) shape
cornell_logs = pickle.load(open(os.path.join(MAIN_DATASET_PATH, 'cornell_25000.p'), "rb"))

n_candidate = 10
n_last_chat = 3

RANK_SCORE = []
time_test = 50
for i in range(time_test):
    print('-------------- PROCESS %d/%d ------------' % (i, time_test))
    idx = np.random.randint(100, 48000)
    if idx % 2 != 0:
        idx += 1

    # get query
    query = cornell_logs[idx]
    # get response
    response = cornell_logs[idx + 1]
    # get context
    context = cornell_logs[idx - 2: idx + 1]

    # set of candidate
    random_idx = idx + 1000
    candidates = cornell_logs[random_idx: random_idx + n_candidate]
    candidates.append(response)
    shuffle(candidates)

    results = Rank(query, candidates, context)

    # Check: (1 in 10 R@10, 1 in 10 R@5, 1 in 10 R@1)
    for j, item in enumerate(results):
        if j < 5 and item[0] == response:
            break
    if j < 1:
        RANK_SCORE.append(0)
    elif j < 2:
        RANK_SCORE.append(2)
    elif j < 5:
        RANK_SCORE.append(5)
    else:
        RANK_SCORE.append(9)

sum([x == 0 for x in RANK_SCORE])
