import pandas as pd
import processText
import nltk

stop_words = nltk.corpus.stopwords.words('english')

new_sw = []
new_sw = [word.replace("'", "") for word in stop_words if word.find("'") > -1]

stop_words.extend(new_sw)

more_sw = ['im', 'id', 'hed', 'shed', 'hes', 'shes', 'theyre', 'ive']
stop_words.extend(more_sw)

def process(text, stop_words):
    sentences = processText.tokenize_sentences(text)
    term_vec = processText.tokenize_words(sentences)
    term_vec = processText.remove_stop_words(term_vec, stop_words)
    norm = processText.rebuild(term_vec)
    norm = processText.remove_empty_terms(norm)
    return norm

trump = pd.read_csv('trump.csv', encoding = "ISO-8859-1")
norm_trump = process(trump['text'], stop_words)
print(norm_trump)
