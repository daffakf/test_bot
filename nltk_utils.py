import numpy as np
import nltk

# Porter Stemmer
from nltk.stem.porter import PorterStemmer
# stemmer = PorterStemmer()

# PySastrawi
# pip install PySastrawi
# import StemmerFactory class
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# nltk.download('punkt') # pre-trained tokenize, 1-time run

# Stopwords
# nltk.download('stopwords')
# from nltk.corpus import stopwords

def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["menjadi", "terjadi", "kejadian"]
    words = [stem(w) for w in words]
    -> ["jadi", "jadi", "jadi"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag

# Tokenization test
# a = "your conTact details pleaSe?"
# print(a)
# a = tokenize(a)
# print(a)

# BoW test
# sentence = ["your", "contact", "detail"]
# words = ["can", "contact", "detail", "how", "your"]
# bog = bag_of_words(sentence, words)
# print(bog)

# stemming process
# kalimat = 'Perekonomian Indonesia sedang dalam pertumbuhan yang membanggakan'
# output   = stemmer.stem(kalimat)

# print(output)
# ekonomi indonesia sedang dalam tumbuh yang bangga

# print(stemmer.stem('Mereka menyerukan'))
# mereka tiru