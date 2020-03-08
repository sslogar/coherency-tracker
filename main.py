import pandas as pd
import processText
import nltk
import coherency

stop_words = nltk.corpus.stopwords.words('english')

new_sw = [word.replace("'", "") for word in stop_words if word.find("'") > -1]

stop_words.extend(new_sw)

more_sw = ['im', 'id', 'hed', 'shed', 'hes', 'shes', 'theyre', 'ive', 'one', 'like']
stop_words.extend(more_sw)
# print(stop_words)

# def extend_sw(sw, arr=[]):
#     if len(arr) > 0:
#         sw.extend(arr)
#     else:
#         new_sw = new_sw = [word.replace("'", "") for word in sw if word.find("'") > -1]
#         sw.extend(new_sw)
#     return sw

# stop_words = extend_sw(stop_words)

def process(text, stop_words):
    sentences = processText.tokenize_sentences(text)
    term_vec = processText.tokenize_words(sentences)
    term_vec = processText.remove_stop_words(term_vec, stop_words)
    norm = processText.rebuild(term_vec)
    norm = processText.remove_empty_terms(norm)
    return norm

trump = pd.read_csv('trump.csv', encoding = "ISO-8859-1")
norm_trump = process(trump['text'], stop_words)

tokenized = coherency.tokenizeCorpus(norm_trump)
m1 = coherency.w2v(tokenized, size=10, window=5, count=1, sample=1e-3)
print(coherency.returnSimilarWords(m1, ['red', 'ukraine', 'omar', 'democrats', 'great', 'america', 'congress']))
# print(norm_trump)
