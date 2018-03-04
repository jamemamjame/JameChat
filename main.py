'''
Overall of project
1) get query from user
2) clean

@author: Jame Phankosol
'''


from src import filepath as path
from response_retrival.retrival import retrival

# assume the user query
query = 'Do you know the history of Beijing?'
# name of document file
doc_name = path.DOCUMENT

retrival = retrival(filename=doc_name)
candidates = retrival.retrive(query)

for candidate in candidates:
    rank(query, candidate)