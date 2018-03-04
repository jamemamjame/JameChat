'''
Retrival Method
score by most shared word
'''
from sentence_prepocessing import Sentence_prepocess


class retrival():
    def __init__(self, filename, stemer='Porter', n_candidate=10):
        '''
        :param filename: path of file which abundant with sentence 
        :param stemer: 
        :param n_candidate: 
        '''
        self.__sentence_proc = Sentence_prepocess(stemer=stemer,
                                                  keep_stopword=False)  # ควรตัดให้ cleaning == clean == cleaned == cleans
        # file of a sentence in document
        self.__filename = filename
        self.__n_candidate = n_candidate

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
        :return: set of candidate response
        '''
        # list of (response, score) that have top score
        poss_reponse = [('', -100 + i) for i in range(0, self.__n_candidate)]

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
                        poss_reponse.insert(i, (line, score))
                        poss_reponse.pop(0)

                if i == len(poss_reponse) - 1:
                    poss_reponse.append((line, score))
                    poss_reponse.pop(0)

                # read new line
                line = f.readline().strip()

        return [candidate for (candidate, _) in poss_reponse]


# rtv = retrival(filename='./src/ubuntu_logs.txt')
# a = rtv.retrive('Hello. Which command is to install audio and video codecs?')
