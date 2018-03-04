'''
Overall of project
1) get query from user
2) clean

@author: Jame Phankosol
'''
from sentence_prepocessing import Sentence_prepocess

# sentence_proc = Sentence_prepocess(stemer='Porter')


from retrival import retrival

# assume the user query
query = 'good morning :)'
# name of document file
doc_name = './src/ubuntu_logs.txt'

retrival = retrival(filename=doc_name)
candidate = retrival.retrive(query)
