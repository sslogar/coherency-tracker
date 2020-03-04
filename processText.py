import pandas as pd
import nltk
from gensim.models import Word2Vec
import re
import string
from nltk.tokenize import sent_tokenize, word_tokenize

def tokenize_sentences(text):
    # sub_s = sent_tokenize(line) for line in text
    # sentences = [replace_chars(s) for s in sub_s]
    sentences = []
    for line in text:
        sub_s = sent_tokenize(line)
        for s in sub_s:
            s = replace_chars(s)
            sentences.append(s)
    return sentences

def rebuild(term_vec):
    for i in range(0, len(term_vec)):
        doc = ""
        doc = ' '.join(term_vec[i])
        term_vec[i] = doc
    return term_vec

def replace_chars(s):
    s = s.replace('\x92', "'")
    s = s.replace('\x93', '\"')
    s = s.replace('\x94', '\"')
    return s

def tokenize_words(doc):
    #tokenize into words
    punc = re.compile('[%s]' % re.escape(string.punctuation))
    term_vec = []
    for d in doc:
        d=d.lower()
        d= punc.sub('', d)
        term_vec.append(word_tokenize(d))
    return term_vec

def remove_stop_words(term_vec, sw):
    for i in range(0, len(term_vec)):
        term_list = []
        for term in term_vec[i]:
            if term not in sw:
                term_list.append(term)
        term_vec[i] = term_list
    return(term_vec)

# model = Word2Vec(norm_trump, size=100, window=5, min_count=1, workers=4)
# print(model)
