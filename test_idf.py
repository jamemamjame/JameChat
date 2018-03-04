from numpy import log


class idf():
    def __init__(self, all_document):
        # self.all_document = all_document
        self.idf_representation = self.inverse_document_frequencies(all_document)

    def inverse_document_frequencies(self, all_document):
        '''
        This method will find a idf_representation from a given all_document

        IDF = inverse_document_frequencies
        at here -> 'word' == 'token'

        IDF-equation from https://en.wikipedia.org/wiki/Tf–idf
        :param all_document:
        :return:
        '''
        idf_values = {}
        all_tokens_set = set([word for subdoc in all_document for word in subdoc])
        for tkn in all_tokens_set:
            contains_token = map(lambda subdoc: tkn in subdoc, all_document)
            idf_values[tkn] = log(len(all_tokens_set) / float(1 + sum(contains_token)))
        return idf_values

    def get_idf(self, word):
        return self.idf_representation[word]


from numpy import array

tokenize = lambda doc: doc.lower().split(" ")

document_0 = "China has a strong economy that is growing at a rapid pace. However politically it differs greatly from the US Economy."
# document_1 = "At last, China seems serious about confronting an endemic problem: domestic violence and corruption."
# document_2 = "Japan's prime minister, Shinzo Abe, is working towards healing the economic turmoil in his own country for his view on the future of his people."
# document_3 = "Vladimir Putin is working hard to fix the economy in Russia as the Ruble has tumbled."
# document_4 = "What's the future of Abenomics? We asked Shinzo Abe for his views"
# document_5 = "Obama has eased sanctions on Cuba while accelerating those against the Russian Economy, even as the Ruble's value falls almost daily."
# document_6 = "Vladimir Putin is riding a horse while hunting deer. Vladimir Putin always seems so serious about things - even riding horses. Is he crazy?"

# all_documents = [document_0, document_1, document_2, document_3, document_4, document_5, document_6]
all_documents = [document_0]
all_docs = array([tokenize(doc) for doc in all_documents])
myidf = idf(all_docs)
