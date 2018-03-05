'''
Ranking
-------

Rank(S,Q)= Zigma[ λk · hk(S,Q) ]
'''

# Importing the library
import response_ranking.word_features.word_to_vec as W2V
import response_ranking.word_features.word_matching as WM


def rank(query, candidates):
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
        score = W2V.get_score(query, candidate) + WM.get_score(query, candidate, candidates)
        results.append((candidate, score))

    # sorted a scores list by key = score
    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results


'''
ต้องเอา Q ไปเทสกับฟีเจอร์ต่างๆ แล้วนำคะแนนที่ได้มาใส่เป็นเวกเตอร์ แล้วนำไปโยนให้โมเดลอีกตัวทำนาย
'''
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
# results = rank(query, candidates)