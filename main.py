import pandas as pd
import processText
import nltk
import coherency
import topicModel
import glob
import os

stop_words = nltk.corpus.stopwords.words('english')

new_sw = [word.replace("'", "") for word in stop_words if word.find("'") > -1]

stop_words.extend(new_sw)

more_sw = ['im', 'id', 'hed', 'shed', 'hes', 'shes', 'theyre', 'ive', 'one', 'like', 'sic']
stop_words.extend(more_sw)

def process(text, stop_words):
    sentences = processText.tokenize_sentences(text)
    term_vec = processText.tokenize_words(sentences)
    term_vec = processText.remove_stop_words(term_vec, stop_words)
    norm = processText.rebuild(term_vec)
    norm = processText.remove_empty_terms(norm)
    return norm

def get_files(path):
    all_files = glob.glob(os.path.join(path, "*.csv"))
    all_df = [pd.read_csv(file, encoding = "ISO-8859-1") for file in all_files]
    return all_df

def create_dataset(text_dfs, path):
    name=os.path.basename(path)

    text_dic = {name+str(text['year'])+str(idx): process(text['text'], stop_words) for (idx, text) in enumerate(text_dfs)}
    new_keys = {}

    for (idx, key) in enumerate(text_dic.keys()):
        year = [int(s) for s in key.split() if s.isdigit()][0]
        new_keys[key] = (name + str(idx)+ 'year'+str(year))

    for key, value in new_keys.items():
        text_dic[value] = text_dic.pop(key)

    return text_dic

def run_model(d):
    results = {}
    for key, value in d.items():
        word_dict, word_corpus = processText.create_dictionary_and_corpus(value)
        c = topicModel.getCoherency(word_dict, word_corpus, 10, 'u-mass', varyTopics=True)
        print((key, c))
        results[key] = c
    return results

path = r'C:\\Users\\xruns\\Documents\\Python Scripts\\Coherency\\datasets\\trump'
text_dfs = get_files(path)

text_dic = create_dataset(text_dfs, path)
# print(text_dic)

coherencies = run_model(text_dic)
#
# word_dict, word_corpus = processText.create_dictionary_and_corpus(norm_text)
# tokenized = coherency.tokenizeCorpus(norm_text)
# m1 = coherency.w2v(tokenized, size=10, window=5, count=1, sample=1e-3)
# print(m1.wv.vocab)
# feature_array = coherency.average_word_vectorizer(tokenized, m1, 10)
# print(topicModel.getCoherency(word_dict, word_corpus, 10, 'u-mass', varyTopics=True))
