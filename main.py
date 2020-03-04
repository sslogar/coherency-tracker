import pandas as pd
import processText
import nltk

stop_words = nltk.corpus.stopwords.words('english')

new_sw = []
for word in stop_words:
    if word.find("'") > -1:
        new_sw.append(word.replace("'", ""))

stop_words.extend(new_sw)
more_sw = ['im', 'id', 'hed', 'shed', 'hes', 'shes', 'theyre']
stop_words.extend(more_sw)

trump = pd.read_csv('trump.csv', encoding = "ISO-8859-1")

sentences = processText.tokenize_sentences(trump['text'])
term_vec = processText.tokenize_words(sentences)
# print(term_vec)
term_vec = processText.remove_stop_words(term_vec, stop_words)
# print(term_vec)
norm_trump = processText.rebuild(term_vec)
# print(norm_trump)
