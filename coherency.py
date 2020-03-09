from gensim.models import Word2Vec
from nltk.tokenize import WordPunctTokenizer
import numpy as np

def tokenizeCorpus(text):
    return [WordPunctTokenizer().tokenize(doc) for doc in text]

def w2v(text, size, window, count, sample):
    model = Word2Vec(text, size=size, window=window,
            min_count=count, workers=4, sample=sample, iter=50)
    return model

def returnSimilarWords(model, searchTerms):
    similar = {search_term: [item[0] for item in model.wv.most_similar([search_term], topn=5)]
                for search_term in searchTerms}
    return similar

def average_word_vectors(words, model, vocab, nfeatures):
    feature_vector = np.zeros((nfeatures,), dtype='float64')
    nwords = 0.
    for word in words:
        if word in vocab:
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model[word])
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)
    return feature_vector

def average_word_vectorizer(text, model, nfeatures):
    vocab = set(model.wv.index2word)
    features = [average_word_vectors(sentence, model, vocab, nfeatures) for sentence in text]
    return np.array(features)
