from nltk.tokenize import WordPunctTokenizer
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import numpy as np
from gensim.corpora import Dictionary

def getCoherency(d, corp, topics=10, coherence='u-mass', varyTopics=False):
    m1 = LdaModel(corp, topics, d)
    cm = CoherenceModel(model=m1, corpus=corp, coherence='u_mass')
    if varyTopics:
        topics = range(5, 16)
        coherencies = []
        for topic in topics:
            m = LdaModel(corp, topic, d)
            c = CoherenceModel(model=m, corpus=corp, coherence='u_mass')
            coherencies.append(c.get_coherence())
        return np.max(coherencies)
    return cm.get_coherence()
