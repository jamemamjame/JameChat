'''
Prepare a word sequence (sentence) with padding
    - convert a sentence String to list of words in sentence
    - convert list of words in sentence to array of word_embedding and padded
'''
import numpy as np
from keras.preprocessing import sequence
from textblob import Sentence
from src.static_variable import load_word_embedding

# Example dialogs
dialogs = [
    "Beijing is a historical city that can be traced back to 3,000 years ago.",
    "The city's history dates back three millennia. As the last of the Four Great Ancient Capitals of China",
    "Beijing has been the political center of the country for much of the past eight centuries",
    "With mountains surrounding the inland city on three sides"
]

# load pre-trained Word Embedding model
WORD_EMB = load_word_embedding(load_glove=True)

# constant of maximum number of word in sentence (important when padding)
MAX_WORD = 20

def prep_text(txt):
    '''
    Process a text string to list of embedded word
    'I love you' -> ['I', 'love', 'you'] -> list([emb('I'), emb('love'), emb('you')])
    :param word_seq: text String
    :return: list of embedded word
    '''
    word_seq = Sentence(txt.lower()).words
    wordlist = []
    for word in word_seq:
        if word in WORD_EMB.vocab:
            wordlist.append(WORD_EMB[word])
        else:
            # shape is depend on
            wordlist.append(np.full(shape=[100], fill_value=0.0, dtype=np.float64))
    return wordlist

def generate_dialogs_embedding(dialogs):
    tmp_dialogs = []
    for sentence in dialogs:
        word_emb_list = prep_text(sentence)
        tmp_dialogs.append(word_emb_list)

    # padding a word that lower than max_words
    return sequence.pad_sequences(tmp_dialogs, maxlen=MAX_WORD, dtype=np.float64)

