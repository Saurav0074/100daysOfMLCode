import gensim
from gensim import corpora as corpora
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from matplotlib import pyplot as plt

import  pickle
from text_prediction import split_text_to_sentences
from extract_html import get_texts_from_xmls, get_qa_pairs_from_xmls
import pyLDAvis.gensim
from shutil import  copyfile

import csv, re
import spacy
from spacy.lang.en import  English
import extract_prediction_files
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import  os, random
import numpy as np
import  pandas as pd
from nltk.corpus import  wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from pprint import  pprint

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

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def compute_coherence_values(dictionary, corpus, texts, start, limit, step):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    print("Finding optimal # topic")
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        print("Doing" + str(num_topics) + "no. of topics")
        model = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


def write_dataframe_to_files():
    df = pickle.load(open('dataframe', 'rb'))

    print(df.head(10))
    #data = df.

def assign_topics_to_sentences(ldamodel, corpus, texts ):

    main_df = pd.DataFrame()

    # get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0: # 0th is the dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                main_df = main_df.append(pd.Series([int(topic_num), topic_keywords]),
                                         ignore_index=True)

    contents = pd.Series(texts)
    main_df = pd.concat([main_df, contents], axis=1)

    main_df.columns = ['dominant_topic', 'Keywords', 'Text']
    pickle.dump(main_df, open('blogs_df', 'wb'))

    pprint(main_df.head(10))
    return  main_df


if __name__ == "__main__":

    file = 'new_female_corpus.txt'
    src_dir = 'C:\\Users\\sjha\\Documents\\Subject_tagging\\' + file

    texts = read_blogs(src_dir)
    print(len(texts))

    texts = texts[:20000]

    print("Loaded texts .. ")
    preprocessed_texts = do_preprocessing(texts)

    data_words = list(sent_to_words(preprocessed_texts))

    data_words_nostops = remove_stopwords(data_words)

    # create bigrams and trigrams
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    print("Creating bigrams")

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]


    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    data_words_bigrams = make_bigrams(data_words_nostops)

    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # keep only noun, adj, verb, adv after lemmatization
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])


    # creating dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    print("Creating dictionary .. ")
    # create corpus
    texts = data_lemmatized

    # term-doc frequency
    corpus = [id2word.doc2bow(text) for text in texts]

   # Finding the optimal no of topics

    model_list, coherence_values = compute_coherence_values(id2word, corpus, data_lemmatized, 15, 90, 15)

    pickle.dump(model_list, open('model_list','wb'))
    pickle.dump(coherence_values, open('coherence_values', 'wb'))
    limit = 90
    start = 15
    step = 15

    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("#Topics")
    plt.ylabel("Coherence values")
    plt.legend(('coherence_values'), loc='best')
    plt.show()
    plt.savefig('coherence_graph.png')

    # mallet_path = r'C:\Users\sjha\Documents\mallet-2.0.8\bin\mallet'
    # ldamallet = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word=id2word, passes=15)
    # # ldamallet.save('model_blog.gensim')
    # ldamallet.load('model_blog.gensim')
    #
    #
    # # outputs = ldamallet.show_topics(formatted=False)
    # topics = ldamallet.print_topics(num_words=10)
    #
    # for topic in topics:
    #     pprint(topic)

