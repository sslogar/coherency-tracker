from gensim.models import Word2Vec

model = Word2Vec(text, size=100, window=5, min_count=1, workers=4)
print(model)
