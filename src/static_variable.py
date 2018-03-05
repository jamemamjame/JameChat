'''
File Path Server
'''
from nltk.corpus import stopwords

# # # # # # # # # # # # # FILE PATH # # # # # # # # # # # # #
# path of google word2vec
PATH_GOOGLE_WORD2VEC = '/Users/jamemamjame/Computer-Sci/_chula course/SENIOR PROJECT/Resource/GoogleNews-vectors-negative300.bin'
# path of sentence corpus
PATH_DOCUMENT = './src/beijing.txt'


NOUN, VERB, ADJ, ADV = 'n', 'v', 'j', 'r'
POSMAP = {'N': NOUN, 'V': VERB, 'j': ADJ, 'r': ADV}
STOPWORDS = set(stopwords.words('english'))