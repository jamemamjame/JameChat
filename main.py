'''
Overall of project
1) get query from user
2) clean

@author: Jame Phankosol
'''


from src import filepath as path
from response_retrival.retrival import retrive as Retrive
from response_ranking.ranking import rank as Rank

# assume the user query
query = 'Do you know the history of Beijing?'


candidates = Retrive(query)

for candidate in candidates:
    Rank(query, candidate)