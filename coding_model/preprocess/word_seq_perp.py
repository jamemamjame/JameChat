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
    "With mountains surrounding the inland city on three sides",
    "When you want to use a different language, select the same Keyboard & Character Viewer item in the menu bar -- although, when more than one language is enabled, the icon will now show the flag or symbol of the current language in use -- and click on the desired language. The Keyboard Viewer will then change layout to represent the keys of the desired language and allow you to insert native characters. To change back to your regular language, once again click on the Keyboard & Character Viewer menu bar item and select the desired input language to switch to."
]

# load pre-trained Word Embedding model
WORD_EMB = load_word_embedding(load_glove=False)

# constant of maximum number of word in sentence (important when padding)
MAX_WORD = 11
N_CHANEL = 1


def generate_wordlist_emb(txt):
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
            # shape is depend on embedding dimension
            # use [1., 1., 1., ...] represent a unknown word
            wordlist.append(np.full(shape=[300], fill_value=1.0, dtype=np.float32))
    return wordlist


def generate_sentence_embedding(sentence):
    # get list of embedded words
    new_sentence = generate_wordlist_emb(sentence)

    # padding a sentence by add 0 value at the front until sentence's length = max_word
    return sequence.pad_sequences([new_sentence], maxlen=MAX_WORD, dtype=np.float32)[0]


def generate_dialogs_embedding(dialogs):
    tmp_dialogs = []
    for i, sentence in enumerate(dialogs):
        tmp_dialogs.append(generate_sentence_embedding(sentence))

    # padding a word that lower than max_words
    tmp_dialogs = np.array(tmp_dialogs)

    # n_sample, max_word, emb_dim, = tmp_dialogs.shape[0], tmp_dialogs.shape[1], tmp_dialogs.shape[2]

    # # return with add 1 chanel in the last index
    # return np.reshape(tmp_dialogs, [n_sample, max_word, emb_dim, N_CHANEL])
    return tmp_dialogs

# dialogs_emb = generate_dialogs_embedding(dialogs)
