'''
Retrival Method
score by most shared word
'''
from numpy import inf
from sentence_prepocessing import Sentence_prepocess


class retrival():
    query = None
    def __init__(self, filename, stemer='Porter', num_top_response=10):

        self.__sentence_proc = Sentence_prepocess(stemer=stemer,
                                                  keep_stopword=False)  # ควรตัดให้ cleaning == clean == cleaned == cleans
        # file of a sentence in document
        self.__filename = filename
        self.__num_top_response = num_top_response

    def __get_score(self, query, res):
        '''
        :param query: list of pure word
        :param res: list of pure word
        :return:
        '''
        # convert list() to set() because we need to get only unique word for compare
        return len(set(query).intersection(set(res))) / (len(query) + len(res))

    def retrive(self, query):
        '''
        Retrieve a set of sentence that have a chance to be a best response given query
        :param query: String of query (question)
        :return: set of possible response
        '''
        min_score = -inf
        min_idx = 0

        poss_reponse = [('', -100 + i) for i in range(0, self.__num_top_response)]

        with open(self.__filename) as f:
            line = f.readline().strip()
            while True:
                if line == '':
                    break

                tmp_line = self.__sentence_proc.cleaning(line)
                score = self.__get_score(query=query, res=tmp_line)

                for i, (st, st_score) in enumerate(poss_reponse):
                    if i == 0 and score <= st_score:
                        break
                    if score > st_score:
                        continue
                    if score <= st_score:
                        poss_reponse.insert(i, (tmp_line, score))
                        poss_reponse.pop(0)

                if i == len(poss_reponse) - 1:
                    poss_reponse.append((line, score))
                    poss_reponse.pop(0)

                # read new line
                line = f.readline().strip()

# __sentence_proc = Sentence_prepocess(stemer='Porter',
#                                      keep_stopword=False)  # ควรตัดให้ cleaning == clean == cleaned == cleans

# def __get_score(query, res):
#     '''
#     :param query: list of pure word
#     :param res: list of pure word
#     :return:
#     '''
#     # convert list() to set() because we need to get only unique word for compare
#     return len(set(query).intersection(set(res))) / (len(query) + len(res))
#
#
# # file of a sentence in document
# filename = './src/ubuntu_logs.txt'
# query = 'how do i restore a file that i accidentally deleted from an .deb package?'
# query = __sentence_proc.cleaning(query)
#
# num_top_response = 5

# poss_reponse = [('', -100 + i) for i in range(0, num_top_response)]
# min_score = -inf
# min_idx = 0

with open(filename) as f:
    line = f.readline().strip()
    while True:
        if line == '':
            break
        tmp_line = __sentence_proc.cleaning(line)
        score = __get_score(query=query, res=tmp_line)

        for i, (st, st_score) in enumerate(poss_reponse):
            if i == 0 and score <= st_score:
                break
            if score > st_score:
                continue
            if score <= st_score:
                poss_reponse.insert(i, (tmp_line, score))
                poss_reponse.pop(0)

        if i == len(poss_reponse) - 1:
            poss_reponse.append((line, score))
            poss_reponse.pop(0)

        # read new line
        line = f.readline().strip()
