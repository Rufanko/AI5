import numpy as np
import math
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



class NaiveBayesClassifier:
    def __init__(self):
        self.v = 0 
        self.d = 0
        self.length = 0
        self.stat = dict()
        
    def fit(self, word_mat, b_vec):
        uniq, count = np.unique(b_vec, return_counts=True)
        length = len(word_mat[0])
        self.length = length
        self.stat = { uniq[i] : (count[i], 0, np.array([0] * length)) for i in range(len(uniq)) }
        self.d = count.sum()
        vocabulary = np.array([0] * length)
        for i in range(len(b_vec)):
            temp = self.stat[b_vec[i]]
            self.stat[b_vec[i]] = (temp[0],
                temp[1] + word_mat[i].sum(), temp[2] + word_mat[i])
            vocabulary += word_mat[i]
        self.v = np.count_nonzero(vocabulary)

    def predict(self, word_mat):
        classes = []
        for j in range(len(word_mat)):
            max_value = -math.inf
            c_value = None
            for c in self.stat.keys():
                result = math.log(self.stat[c][0] / self.d)
                for i in range(self.length):
                    if word_mat[j][i]:
                        term = math.log(
                            (self.stat[c][2][i] + 1) /
                            (self.v + self.stat[c][1]))
                        result += word_mat[j][i] * term
                if result > max_value:
                    max_value = result
                    c_value = c
            classes.append(c_value)
        return classes


def stemmer(texts):
    result = []
    ps = PorterStemmer()
    for text in texts:
        words = text.lower().split()
        stop = set(stopwords.words('english'))
        stem_word = [ps.stem(word) for word in words if not word in stop]
        result.append(' '.join(stem_word))
    return result

def to_arr():
    neg_indexes = movie_reviews.fileids('neg')
    pos_indexes = movie_reviews.fileids('pos')
    neg_reviews = [movie_reviews.raw(fileids=ids) for ids in neg_indexes]
    pos_reviews = [movie_reviews.raw(fileids=ids) for ids in pos_indexes]
    list_of_text = stemmer(neg_reviews + pos_reviews)
    cv = CountVectorizer()
    word_arr = cv.fit_transform(list_of_text).toarray()
    bin_arr = np.array([0] * len(neg_reviews) + [1] * len(pos_indexes))
    return (word_arr, bin_arr)


nltk.download('movie_reviews')
nltk.download('stopwords')
word_arr, bin_arr = to_arr()
x_train, x_test, y_train, y_test = train_test_split(word_arr, bin_arr, test_size = 0.3, random_state = 0, shuffle = True)

my_class = NaiveBayesClassifier()
my_class.fit(x_train, y_train)
pred = my_class.predict(x_test)
print('My implementation accuracy score = {0}\n'.format(accuracy_score(y_test, pred)))

lib_class = MultinomialNB()
lib_class.fit(x_train, y_train)
pred = lib_class.predict(x_test)
print('Lib implementation accuracy_score = {0}\n'.format(accuracy_score(y_test, pred)))
