import pandas as pd
import nltk
from gensim.models import word2vec
import re
import string
from nltk.tokenize import sent_tokenize, word_tokenize

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

stop_words = nltk.corpus.stopwords.words('english')
print(stop_words)

new_sw = []
for word in stop_words:
    if word.find("'") > -1:
        new_sw.append(word.replace("'", ""))

stop_words.extend(new_sw)

trump = pd.read_csv('trump.csv', encoding = "ISO-8859-1")

sentences = []
for line in trump['text']:
    sub_s = sent_tokenize(line)
    for s in sub_s:
        s = replace_chars(s)
        sentences.append(s)
# print(sentences)

term_vec = tokenize_words(sentences)
# print(term_vec)
term_vec = remove_stop_words(term_vec, stop_words)
# print(term_vec)
norm_trump = rebuild(term_vec)
print(term_vec)
