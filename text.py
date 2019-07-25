import gensim
from gensim.models.keyedvectors import KeyedVectors
path = r"./glove.6B.50d.txt.w2v"
glove = KeyedVectors.load_word2vec_format(path, binary=False)

def se_text(caption):
    wordlist = caption.split()
    vector = 0
    for word in wordlist:
        vector += glove[word]
    #idf and normalize?
        