'''
Overall of project
1) get query from user
2) clean

@author: Jame Phankosol
'''

from src import static_variable as path
from response_retrival.retrival import Retrieve as Retrive
from response_ranking.ranking import Rank as Rank

# assume the user query
query = 'Do you know the history of Beijing?'

candidates = Retrive(query)

for candidate in candidates:
    Rank(query, candidate)
