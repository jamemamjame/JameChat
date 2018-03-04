'''
Lets understand different possible data preprocessing activities:

Convert text to lowercase
    – This is to avoid distinguish between words simply on case.
Remove Number
    – Numbers may or may not be relevant to our analyses. Usually it does not carry any importance in sentiment analysis
Remove Punctuation
    – Punctuation can provide grammatical context which supports understanding. For bag of words based sentiment analysis punctuation does not add value.
Remove English stop words
    – Stop words are common words found in a language. Words like for, of, are etc are common stop words.
Remove Own stop words(if required)
    – Along with English stop words, we could instead or in addition remove our own stop words. The choice of own stop word might depend on the domain of discourse, and might not become apparent until we’ve done some analysis.
Strip white space
    – Eliminate extra white spaces.
Stemming – Transforms to root word. Stemming uses an algorithm that removes common word endings for English words, such as “es”, “ed” and “’s”. For example i.e., 1) “computer” & “computers” become “comput”
Lemmatisation
    – transform to dictionary base form i.e., “produce” & “produced” become “produce”
Sparse terms
    – We are often not interested in infrequent terms in our documents. Such “sparse” terms should be removed from the document term matrix.

cr: https://datascience.stackexchange.com/questions/11402/preprocessing-text-before-use-rnn/11421
'''

# Importing the library
import re  # regular expression
from numpy import array
from nltk.corpus import stopwords


class Sentence_prepocess():
    # regx for substitution string
    __regx_rm_char = None
    __stop_words = set(stopwords.words('english'))

    def __set__stemer(self, stemmer=None):
        if stemmer is None:
            return None
        elif stemmer.lower().__eq__('Porter'.lower()):
            from nltk.stem.porter import PorterStemmer
            return PorterStemmer()
        elif stemmer.lower().__eq__('WordNet'.lower()):
            from nltk.stem.lancaster import WordNetLemmatizer
            return WordNetLemmatizer()
        elif stemmer.lower().__eq__('Lancaster'.lower()):
            from nltk.stem.lancaster import LancasterStemmer
            return LancasterStemmer()
        elif stemmer.lower().__eq__('Snowball'.lower()):
            from nltk.stem.snowball import SnowballStemmer
            return SnowballStemmer()
        else:
            return None

    def __init__(self, stemer=None, keep_stopword=True, regx_rm_char='[^a-zA-Z0-9]'):
        self.__keep_stopword = keep_stopword
        self.__regx_rm_char = regx_rm_char
        self.__stemmer = self.__set__stemer(stemer)

    def cleaning(self, sentence):
        '''
        Convert a simple String to a list of cleaned words
        (still keep stopwords)
        :param sentence: String of sentence
        :return: list of pure words in sentence
        '''

        # Remove a special character
        sentence = self.remove_special_char(sentence)

        # Convert to lowcase
        sentence = sentence.lower()

        # Remove Punctuation
        sentence = sentence.split()

        # Remove English stop words
        if not self.__keep_stopword:
            sentence = self.remove_stopwords(sentence)

        # Remove Own stop words(if required)

        # Stemming
        if self.__stemmer is not None:
            sentence = [self.__stemmer.stem(word) for word in sentence]

        return array(sentence)

    def remove_stopwords(self, sentence):
        return [word for word in sentence if word not in self.__stop_words]

    def remove_special_char(self, sentence):
        return re.sub(pattern=self.__regx_rm_char, repl=' ', string=sentence)

# if __name__ == '__main__':
#     sentence = 'My last name look like your dogs number 13, right? Told me!!!'
#     stp = Sentence_prepocess()
#     stp.cleaning(sentence)
