'''
Retrival
--------
score by most shared non-stopwords
'''
from sentence_prepocessing import Sentence_prepocess
from src import filepath as path

# file of a sentence in document
filename = path.DOCUMENT
n_candidate = 10

# should cut >> cleaning == clean == cleaned == cleans == cleanly
sentence_proc = Sentence_prepocess(stemer='Porter',
                                   keep_stopword=False)


def __get_score(query, res):
    '''
    :param query: list of pure word
    :param res: list of pure word
    :return:
    '''
    # convert list() to set() because we need to get only unique word for compare
    return len(set(query).intersection(set(res))) / (len(query) + len(res))


def retrive(query):
    '''
    Retrieve a set of sentence that have a chance to be a best response given query
    :param query: String of query (question)
    :return: set of candidate response
    '''
    # extract a non-stopword and stem word
    query = sentence_proc.cleaning(query)

    # list of (response, score) that have top score
    poss_reponse = [('', -100 + i) for i in range(0, n_candidate)]

    with open(filename) as f:
        while True:
            # read new line
            line = f.readline().strip()
            added = False

            if line == '':
                break

            tmp_line = sentence_proc.cleaning(line)
            score = __get_score(query=query, res=tmp_line)

            for i, (st, st_score) in enumerate(poss_reponse):
                if i == 0 and score <= st_score:
                    break
                if score > st_score:
                    continue
                if score <= st_score:
                    poss_reponse.insert(i, (line, score))
                    poss_reponse.pop(0)
                    added = True
                    break

            if not added and i == len(poss_reponse) - 1:
                poss_reponse.append((line, score))
                poss_reponse.pop(0)

    return [candidate for (candidate, _) in poss_reponse]

# # # # # # # # # # # # # # # Unit Test # # # # # # # # # # # # # # # # # #
# if __name__ == '__main__':
#     import src.filepath as path
#     rtv = retrival(filename=path.DOCUMENT)
#     query = 'Do you know the history of Beijing?'
#     a = rtv.retrive(query)