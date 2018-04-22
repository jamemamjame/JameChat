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
        score = W2V.get_score(query, candidate) + WM.get_score(query, candidate, candidates) \
                + ST_EMB.get_score(query, candidate) + CT_EMB.get_score(context, candidate)
        results.append((candidate, score))

    # sorted a scores list by key = score
    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results