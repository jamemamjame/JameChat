'''
Ranking
-------

Rank(S,Q)= Zigma[ λk · hk(S,Q) ]
'''

import response_ranking.word_features.word_to_vec2 as w2v

# given query(Q) and candidate(S)
query = 'Do you know the history of Beijing?'
candidate = 'Beijing is a historical city that can be traced back to 3,000 years ago.'

def rank(query, candidate):
    pass
'''
ต้องเอา Q ไปเทสกับฟีเจอร์ต่างๆ แล้วนำคะแนนที่ได้มาใส่เป็นเวกเตอร์ แล้วนำไปโยนให้โมเดลอีกตัวทำนาย
'''

