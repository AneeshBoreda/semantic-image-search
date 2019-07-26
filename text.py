import gensim
from gensim.models.keyedvectors import KeyedVectors
path = r"./glove.6B.50d.txt.w2v"
glove = KeyedVectors.load_word2vec_format(path, binary=False)

import re, string
from collections import Counter
import numpy as np
punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))






def strip_punc(corpus):
    """ Removes all punctuation from a string.

        Parameters
        ----------
        corpus : str

        Returns
        -------
        str
            the corpus with all punctuation removed"""
    return punc_regex.sub('', corpus)

def to_counter(doc):
    """ 
    Produce word-count of document, removing all punctuation
    and making all the characters lower-cased.
    
    Parameters
    ----------
    doc : str
    
    Returns
    -------
    collections.Counter
        lower-cased word -> count"""
    doc = strip_punc(doc).lower()
    token = sorted(doc.split())
    counter = Counter(token)
    return counter

def to_vocab(counters, k=None, stop_words=None):
    """ 
    [word, word, ...] -> sorted list of top-k unique words
    Excludes words included in `stop_words`
    
    Parameters
    ----------
    counters : Iterable[Iterable[str]]
    
    k : Optional[int]
        If specified, only the top-k words are returned
    
    stop_words : Optional[Collection[str]]
        A collection of words to be ignored when populating the vocabulary
    """
        
    if len(counters) > 1:
        #counter = sum(counters, Counter())
        counter=Counter()
        i=0
        for c in counters:
            i+=1
            for word in c:
                if word not in counter:
                    counter[word]=0
                counter[word]+=c[word]
            if i%10000==0:
                 print(i)
    else:
        counter = counters[0]
        
    if stop_words is not None:
        for sw in stop_words:
            del counter[sw]
        
    if k is None:
        return sorted(list(counter.keys()))
    else:
        x = counter.most_common(k)
        return sorted([word for word, ct in x])

def to_idf(vocab, counters):
    n = []
    print('Length of vocab:',len(vocab))
    it=0
    for t in vocab:
        i = 0
        it+=1
        for doc in counters:
            if t in doc:
                i += 1
        n.append(i)
        if it%100==0:
            print(it)
    N = np.full(np.array(n).shape, len(counters))
    return np.log10(N / np.array(n))
    """
    Given the vocabulary, and the word-counts for each document, computes
    the inverse document frequency (IDF) for each term in the vocabulary.
    
    Parameters
    ----------
    vocab : Sequence[str]
        Ordered list of words that we care about.

    counters : Iterable[collections.Counter]
        The word -> count mapping for each document.
    
    Returns
    -------
    numpy.ndarray
        An array whose entries correspond to those in `vocab`, storing
        the IDF for each term `t`: 
                           log10(N / nt)
        Where `N` is the number of documents, and `nt` is the number of 
        documents in which the term `t` occurs.
    """


def save_idf(captions):
    """
    Parameters:
        captions (list of string docs)
    returns:
    """
    counterlist = []
    cnt=0
    for doc in captions:
        counter = to_counter(doc)
        counterlist.append(counter)
        cnt+=1
        if cnt%10000==0:
            print(cnt)
    print('made list of captions')
    vocab = to_vocab(counterlist)
    print('made vocab')
    idf = to_idf(vocab, counterlist)
    print('made idf')
    vocab_dict = {word:index for index,word in enumerate(vocab)}
    return idf, vocab_dict

def se_text(caption, idf, vocab_dict):
    wordlist = strip_punc(caption).lower().split()
    vector = 0
    for word in wordlist:
        if word in glove:
             vector += glove[word] * idf[vocab_dict[word]]
    vector = vector / np.linalg.norm(vector)
    return vector


"""
caption1 = "happy dog"
caption2 = "sad dog"
captions = [caption1, caption2]

idf, vocab_dict = save_idf(captions)
for caption in captions:
    print(se_text(caption, idf, vocab_dict))
"""

