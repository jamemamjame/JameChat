'''
Word Matching
-------------
    word matching feature that counts the number (weighted by the IDF value of each word in S)
    of non-stopwords shared by S and Q.

    * Document here is list of candidates
'''
from textblob import TextBlob


def tfidf():
    pass


# # # # # # # # # # # # # # # Unit Test # # # # # # # # # # # # # # # # # #
# given query(Q), candidate(S) and candidates(D)
query = 'Do you know the history of Beijing?'
candidate = 'Beijing is a historical city that can be traced back to 3,000 years ago.'
candidates = [
    "Beijing is a historical city that can be traced back to 3,000 years ago.",
    "The city's history dates back three millennia. As the last of the Four Great Ancient Capitals of China",
    "Beijing has been the political center of the country for much of the past eight centuries",
    "With mountains surrounding the inland city on three sides"
]
