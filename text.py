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
        counter = sum(counters, Counter())
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
    for t in vocab:
        i = 0
        for doc in counters:
            if t in doc:
                i += 1
        n.append(i)
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










def se_text(caption):
    wordlist = caption.split()
    counter = to_counter(caption)
    vocab = to_vocab([counter])
    idf = to_idf(vocab, counter)
    
    vector = 0
    i = 0
    for word in wordlist:
        vector += glove[word] * idf[i]
        i += 1
    vector = vector / np.linalg.norm(vector)
    return vector

