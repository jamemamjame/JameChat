'''
Word Matching
-------------
    word matching feature that counts the number (weighted by the IDF val- ue of each word in S)
    of non-stopwords shared by S and Q.
'''
from sentence_prepocessing import Sentence_prepocess

sentence_proc = Sentence_prepocess(stemer='Porter', keep_stopword=False)

# given query(Q) and candidate(S)
query = sentence_proc.cleaning('good morning :)')
candidate = sentence_proc.cleaning('good morning')

