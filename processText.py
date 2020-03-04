import pandas as pd
import nltk
from gensim.models import Word2Vec
import re
import string
from nltk.tokenize import sent_tokenize, word_tokenize

def tokenize_sentences(text):
    sub_s = [sent_tokenize(line) for line in text][0]
    sentences = [replace_chars(s) for s in sub_s]
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
    punc = re.compile('[%s]' % re.escape(string.punctuation))
    term_vec = [d.lower() for d in doc]
    term_vec = [punc.sub('', term) for term in term_vec]
    term_vec = [word_tokenize(term) for term in term_vec]
    return term_vec

def remove_stop_words(term_vec, sw):
    for i in range(0, len(term_vec)):
        term_list = []
        # term_list = [term for term in term_vec[i] if term not in sw]
        for term in term_vec[i]:
            if term not in sw:
                term_list.append(term)
        term_vec[i] = term_list
    return(term_vec)

def remove_empty_terms(doc):
    v = [sent for sent in doc if len(sent) > 0]
    return v
