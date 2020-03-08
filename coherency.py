from gensim.models import Word2Vec
from nltk.tokenize import WordPunctTokenizer

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
