'''
Retrival
--------
score by most shared non-stopwords
'''
from src import static_variable as var
from textblob import Sentence, Word
from nltk.corpus import stopwords

# file of a sentence in document
filename = var.PATH_DOCUMENT
n_candidate = 10

# get a dict of POS map
_POSMAP = var.POSMAP

# get english stopwords from nltk-corpus
stopwords = set(stopwords.words('english'))


def prep_text(txt):
    '''
    prepocess a text to list of word
    :param txt: text String
    :return: list of word
    '''
    word_list = []
    postags = Sentence(txt.lower()).pos_tags
    for word, pos in postags:
        if word in stopwords:
            continue

        if pos[0] in _POSMAP.keys():
            word_list.append(Word(word).lemmatize(_POSMAP[pos[0]]))
        else:
            word_list.append(word)
    return word_list


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
    query = prep_text(query)

    # list of (response, score) that have top score
    poss_reponse = [('', -100 + i) for i in range(0, n_candidate)]

    with open(filename) as f:
        while True:
            # read new line
            line = f.readline().strip()
            added = False

            if line == '':
                break

            tmp_line = prep_text(line)
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
# query = 'Do you know the history of Beijing?'
# retrive(query)
