'''
File Path Server
'''

# path of google word2vec
__PATH_GOOGLE_WORD2VEC = '/Users/jamemamjame/Computer-Sci/_chula course/SENIOR PROJECT/Resource/GoogleNews-vectors-negative300.bin'
__PATH_GLOVE = '/Users/jamemamjame/Computer-Sci/_chula course/SENIOR PROJECT/Resource/glove/glove.6B.100d.txt.word2vec'

# path of sentence corpus
PATH_DOCUMENT = '/Users/jamemamjame/Computer-Sci/_chula course/SENIOR PROJECT/Dataset/document/beijing.txt'


def load_stopwords():
    from nltk.corpus import stopwords
    return set(stopwords.words('english'))


def load_word_embedding(load_glove=False):
    from gensim.models import KeyedVectors
    if load_glove:
        # load the Stanford GloVe model
        return KeyedVectors.load_word2vec_format(__PATH_GLOVE, binary=False)
    else:
        # load Google's pre-trained Word2Vec model
        return KeyedVectors.load_word2vec_format(__PATH_GOOGLE_WORD2VEC, binary=True)
