'''
Ranking
-------

Rank(S,Q)= Zigma[ λk · hk(S,Q) ]
'''

# Importing the library
import response_ranking.word_features.word_embedding as W2V
import response_ranking.word_features.word_matching as WM


def Rank(query, candidates):
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
#     "China has a strong economy that is growing at a rapid pace. However politically it differs greatly from the US Economy.",
#     "At last, China seems serious about confronting an endemic problem: domestic violence and corruption.",
#     "Japan's prime minister, Shinzo Abe, is working towards healing the economic turmoil in his own country for his view on the future of his people.",
#     "Vladimir Putin is working hard to fix the economy in Russia as the Ruble has tumbled.",
#     "What's the future of Abenomics? We asked Shinzo Abe for his views",
#     "Obama has eased sanctions on Cuba while accelerating those against the Russian Economy, even as the Ruble's value falls almost daily.",
#     "Vladimir Putin is riding a horse while hunting deer. ",
#     "Vladimir Putin always seems so serious about things - even riding horses. Is he crazy?",
# ]
# results = Rank(query, candidates)
