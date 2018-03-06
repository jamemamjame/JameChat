import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from textblob import Sentence
import gensim
from src.static_variable import load_word_embedding

# load pre-trained Word Embedding model
WORD_EMB = load_word_embedding(load_glove=True)


def generate_words_seq(wordlist):
    tmp_wordlist = []
    for word in wordlist:
        if word in WORD_EMB.vocab:
            tmp_wordlist.append(WORD_EMB[word])
        else:
            tmp_wordlist.append(np.full(shape=[150], fill_value=0.0, dtype=np.float32))
    return np.array(tmp_wordlist)


def prep_text(txt):
    '''
    Process a text string to list of word
    :param txt: text String
    :return: list of word
    '''
    return Sentence(txt).words


seed = 1234
np.random.seed(seed=seed)




dialogs = [
    "Beijing is a historical city that can be traced back to 3,000 years ago.",
    "The city's history dates back three millennia. As the last of the Four Great Ancient Capitals of China",
    "Beijing has been the political center of the country for much of the past eight centuries",
    "With mountains surrounding the inland city on three sides"
]
