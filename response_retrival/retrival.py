'''
Retrieval
--------
score by most shared non-stopwords
'''
from src.static_variable import load_stopwords
from textblob import Sentence, Word

# get set of english stopwords
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
    return word_list


def get_score(query, res):
    '''
    :param query: list of pure word
    :param res: list of pure word
    :return:
    '''
    # convert list() to set() because we need to get only unique word for compare
    return len(set(query).intersection(set(res))) / (len(query) + len(res))


def Retrieve(query, document, n_candidate=5):
    '''
    Retrieve a set of sentence that have a chance to be a best response given query
    :param query: String of query (question)
    :param n_candidate: size of candidate that want to retrieve
    :return: set of candidate response
    '''
    # extract a non-stopword and stem word
    query = prep_text(query)

    # list of (response, score) that have top score
    # this list is ascending sorted
    # the min score at index 0 must lower than -1 * n_candidate
    poss_reponse = [('', -(n_candidate + 5) + i) for i in range(0, n_candidate)]

    for line in document:
        line = line.strip()
        added = False

        # calculate score
        score = get_score(query=query, res=prep_text(line))

        # loop for append (sentence, score) to list like a link-list with sorted by score
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
# results = Retrieve(query)
