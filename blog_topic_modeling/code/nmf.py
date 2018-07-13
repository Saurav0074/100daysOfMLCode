import sys
from nltk.corpus import stopwords
import nltk
import spacy
import  re
from gensim.models import ldamodel
import gensim.corpora
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF
import pickle

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

def read_blogs(filename):
    infile = open(filename, 'r', encoding='utf-8')
    blogs = infile.readlines()
    infile.close()

    return blogs

def do_preprocessing(texts):
    texts = [text for text in texts]
    texts = [re.sub('&#*', '', str(sent)) for sent in texts]

    texts = [re.sub('\s+', ' ', str(sent)) for sent in texts]

    texts = [re.sub("\'", "", str(sent)) for sent in texts]

    return  texts

def sent_to_words(texts):
    for sentence in texts:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True)) # removes punctuations


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

if __name__ == '__main__':
    file = 'new_male_corpus.txt'
    src_dir = 'C:\\Users\\sjha\\Documents\\Subject_tagging\\' + file

    blogs = read_blogs(src_dir)
    blogs = blogs[:50000]

    print("Preprocessing..")
    preprocessed_texts = do_preprocessing(blogs)
    data_words = list(sent_to_words(preprocessed_texts))

    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # keep only noun, adj, verb, adv after lemmatization

    data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    print("done preprocessing.. vectorizing now..")
    n_features = 1000

    data_lemmatized = [' '.join(i) for i in data_lemmatized] # convert the list of lists into list of strings for tfidf

    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df = 2, max_features=n_features, \
                                       stop_words=stop_words)

    tfidf = tfidf_vectorizer.fit_transform(data_lemmatized)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    no_topics = 40

    print("Into matrix factorization..")
    nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, \
              init='nndsvd').fit(tfidf)

    no_top_words = 10 # no. of most contributing words to be visualized
    display_topics(nmf, tfidf_feature_names, no_top_words)
