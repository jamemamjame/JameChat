'''
Overall of project
1) get query from user
2) clean

@author: Jame Phankosol
'''
from sentence_prepocessing import Sentence_prepocess

sentence_proc = Sentence_prepocess(stemer='Porter')

# get the input query
query = 'How old are you?'
query = sentence_proc.cleaning(query)

document = [
    'This is my house :)', 'Nice to meet you', 'Really?', 'Can you help me Mercy', 'I\'m 10 years old', 'Are you okay?'
]

#
for i, sentence in enumerate(document):
    document[i] = sentence_proc.cleaning(sentence)

from retrival import Retrival
retrieve = Retrival()
retrieve.retrieve()

