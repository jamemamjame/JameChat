'''
Ranking
-------

Rank(S,Q)= ∑[ λk · hk(S,Q) ]
'''

# Importing the library
import response_ranking.word_features.word_embedding as W2V
import response_ranking.word_features.word_matching as WM
import response_ranking.sentence_features.sentence_emb as ST_EMB

import response_ranking.context_features.context_emb as CT_EMB


def Rank(query, candidates, context):
    '''
    Rank the sentence in candidates set from high->low score
    :param query: String of query
    :param candidates: list of sentence that we need to rank for get a best response given query
    :return: ranked list of candidate
    '''

    # declare a list of (candidate, score)
    results = []

    # loop for calculate score of each candidate
    for candidate in candidates:
        wm_score = WM.get_score(query=query, candidate=candidate, document=candidates)
        w2v_score = W2V.get_score(query=query, candidate=candidate)
        # st_score = ST_EMB.get_score(query=query, candidate=candidate)
        ct_score = CT_EMB.get_score(context=context, candidate=candidate)

        # score = wm_score + w2v_score + st_score + ct_score
        score = wm_score + w2v_score + ct_score

        results.append((candidate, score))

    # sorted a scores list by key = score
    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results


# # # # # # # # # # # # # # # Unit Test # # # # # # # # # # # # # # # # # #
# given query(Q), candidate(S) and candidates(D)
# query = 'Do you know the history of Beijing?'
# candidates = [
#     "Beijing is a historical city that can be traced back to 3,000 years ago.",
#     "The city's history dates back three millennia. As the last of the Four Great Ancient Capitals of China",
#     "Beijing has been the political center of the country for much of the past eight centuries",
#     "With mountains surrounding the inland city on three sides",
#     "Street food in Thailand brings together various offerings of ready to eat meals",
#     "snacks, fruits and drinks sold by hawkers or vendors at food stalls or food carts on the street side in Thailand",
#     "There is scarcely a Thai dish that is not sold by a street vendor or at a market somewhere in Thailand",
# ]
# context = [
#     "China has a strong economy that is growing at a rapid pace. However politically it differs greatly from the US Economy.",
#     "At last, China seems serious about confronting an endemic problem: domestic violence and corruption.",
#     "Japan's prime minister, Shinzo Abe."
# ]


# # # # # # # # # # # # # # # Unit Test # # # # # # # # # # # # # # # # # #
import os, pickle
import numpy as np

MAIN_DATASET_PATH = '/Users/jamemamjame/Computer-Sci/_chula course/SENIOR PROJECT/Dataset/prep_dialog'

enc_log = pickle.load(open(os.path.join(MAIN_DATASET_PATH, 'train_lambda/enc_log_25000.p'), "rb"))
dec_log = pickle.load(open(os.path.join(MAIN_DATASET_PATH, 'train_lambda/dec_log_25000.p'), "rb"))

n_candidate = 10
n_last_chat = 3

RANK_SCORE = []
for i in range(60):
    print('-------------- PROCESS %d/%d ------------' % (i, 50))
    idx = np.random.randint(100, 20000)

    # get query
    query = enc_log[idx]
    # context
    context = [enc_log[idx - 1], dec_log[idx - 1], query]
    # context = enc_log[idx - n_last_chat - 1: idx + 1]
    # response
    resp = dec_log[idx]
    # set of candidate
    candidates = dec_log[idx + 10: idx + 10 + n_candidate - 1]
    candidates.append(resp)

    results = Rank(query, candidates, context)

    # Check: (1 in 10 R@10, 1 in 10 R@5, 1 in 10 R@1)
    for i, item in enumerate(results):
        if i < 5 and item[0] == resp:
            break
    if i < 1:
        RANK_SCORE.append(1)
    elif i < 2:
        RANK_SCORE.append(2)
    elif i < 5:
        RANK_SCORE.append(5)
    else:
        RANK_SCORE.append(0)

pickle.dump(RANK_SCORE, open(os.path.join(MAIN_DATASET_PATH, 'rankscore_noST.p'), "wb"))
# tmp = pickle.load(open(os.path.join(MAIN_DATASET_PATH, 'rank_score_list2.p'), "rb"))
sum([x == 0 for x in RANK_SCORE])
