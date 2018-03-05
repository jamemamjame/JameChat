'''
Retrival
--------
score by most shared non-stopwords
'''
from src.static_variable import PATH_DOCUMENT, POSMAP, STOPWORDS
from textblob import Sentence, Word

# file of a sentence in document
filename = PATH_DOCUMENT
n_candidate = 5

# get a dict of POS map
_POSMAP = POSMAP


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

        if pos[0] in _POSMAP.keys():
            word_list.append(Word(word).lemmatize(POSMAP[pos[0]]))
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
    # this list is ascending sorted
    # the min score at index 0 must lower than -1 * n_candidate
    poss_reponse = [('', -(n_candidate + 5) + i) for i in range(0, n_candidate)]

    with open(filename, 'r') as f:
        while True:
            # read new line
            line = f.readline()

            if line == '':
                break
            elif line.startswith('#') or line == '\n':
                # check for skip line
                continue

            line = line.strip()
            added = False

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
query = 'Do you know the history of Beijing?'
retrive()