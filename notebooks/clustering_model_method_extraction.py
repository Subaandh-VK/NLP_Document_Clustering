#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import json
import spacy
import warnings
from sklearn.metrics.pairwise import cosine_similarity

from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from pathlib import Path

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
warnings.filterwarnings('ignore')

ROOT_DIR = Path('.').resolve().parents[0].absolute()
DATA_DIR = ROOT_DIR / 'dataset' / 'documents'

# Create a list of stopwords
STOP_WORDS = list(fr_stop) + list(en_stop)
# Punctuations to be removed
PUNCTUATIONS = "!#$%&()*+,-./:;<=>?@[\]^_`{|}~//\\«»°"
# Text preprocessing variables
nlp = spacy.load('fr_core_news_sm')

ROOT_DIR, DATA_DIR


def read_data(count=0):
    """

    :param count: The number of file to read if 0 reads all files
    :return: pandas Dataframe
    """
    file_list = []
    content_list = []

    files = DATA_DIR.glob('*')

    for file in files:
        if count != 0 and count == len(file_list):
            break

        with open(file, 'r') as f:
            j_obj = json.load(f)

            file_list.append(j_obj['filename'])
            content_list.append(j_obj['clauses'])

    df = pd.DataFrame(zip(file_list, content_list), columns=['filename', 'clauses'])

    return df


def process_clauses(text):
    processed_text = ''.join(str(val).lower() for val in text)
    processed_text = ''.join(val for val in processed_text if val not in PUNCTUATIONS)

    return processed_text


def tokenize_string(text):
    """
    spaCy text preprocessing for all the clauses
    :param text: The text of clauses to be processed
    :return: the converted text
    """
    text = process_clauses(text)

    doc = nlp(text)

    doc = [word for word in doc if word.pos_ != 'NUM' and word.pos_ != 'SYM' and word.pos_ != 'X']

    lemma_list = []
    for token in doc:
        lemma_list.append(token.lemma_)

    # Remove stopwords
    converted_text = []
    for word in lemma_list:
        lexeme = nlp.vocab[word]
        if lexeme.is_stop == False:
            converted_text.append(word)

            # Remove punctuation digits and empty characters
    converted_text = ' '.join(
        word for word in converted_text if word not in PUNCTUATIONS and word != ' ' and len(word) > 1)

    return converted_text


def vectorize_tfidf(clauses):
    """ Function to vectorize the clauses of the content inside each of the document """

    vectorizer = TfidfVectorizer(max_df=0.7, min_df=0.2)
    vectorizer = vectorizer.fit(clauses)
    X = vectorizer.transform(clauses)

    return X, vectorizer


def train_kmeans(k, X):
    """

    :param k: number of clusters
    :param X: The vectorized clauses
    :return: the model
    """
    km = KMeans(n_clusters=k)
    km = km.fit(X)

    return km


def print_clusters(km, df, feature_names, words, k):
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    for i in range(k):
        print('Cluster %d:' % i),
        for j in order_centroids[i, :words]:
            print(feature_names[j], ',', end=' ')
            print('\n')
    for f in df[df['cluster'] == i]['filename']:
        print(f, ',', end=' ')

    print('\n\n')


def similarity_dict(sp):
    """

    :param sp: pairwise similarity matrix
    :return: dictionary sorted by sum of cosine similarity
    """
    cosine_dict = {}
    idx = 0

    for i in sp:
        cosine_dict[idx] = np.nansum(i)
        idx += 1

    sorted_cosine = {k: v for k, v in sorted(cosine_dict.items(), key=lambda item: item[1])}

    return sorted_cosine


def get_outlier_idx(threshold, similarity_matrix):
    """

    :param threshold: Get outliers below specified threshold
    :param similarity_matrix: the similarity matrix
    :return:
    """
    outlier_idx = []

    for keys, values in similarity_matrix.items():
        if values < threshold:
            outlier_idx.append(keys)

    return outlier_idx


from gensim import corpora, models, similarities
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel


# In[361]:


# Topic Modelling 
def remove_stopwords_from_string(text):
    """ Removes stopwords from a string """
    text = ' '.join([word for word in text.split() if word not in STOP_WORDS])

    return text


def remove_nouns(text):
    doc = nlp(text)
    processed_text = [word for word in doc if word.pos_ != 'PROPN' and word.tag_ != 'NNP']
    processed_text = ' '.join(str(word) for word in processed_text)

    return processed_text


def tokenize_string_topic_modelling(text):
    """ Function to tokenize, remove stopwords, punctuations and stemming """

    text = process_clauses(text)

    doc = nlp(text)

    doc = [word for word in doc if word.pos_ != 'NUM' and word.pos_ != 'SYM' and word.pos_ != 'X']

    lemma_list = []
    for token in doc:
        lemma_list.append(token.lemma_)

    # Remove stopwords
    converted_text = []
    for word in lemma_list:
        lexeme = nlp.vocab[word]
        if lexeme.is_stop == False:
            converted_text.append(word)

            # Remove punctuation digits and empty characters
    converted_text = [word for word in converted_text if word not in PUNCTUATIONS if
                      not word.isdigit() and word != ' ' and len(word) > 1]

    return converted_text


def topic_modelling_data_preprocessing(clauses):
    processed_clause = [remove_nouns(clause) for clause in clauses]
    processed_clause = [tokenize_string_topic_modelling(clause) for clause in processed_clause]
    dictionary = corpora.Dictionary(processed_clause)

    # remove extremes (similar to the min/max df step used when creating the tf-idf matrix)
    dictionary.filter_extremes(no_below=1, no_above=0.8)

    # convert the dictionary to a bag of words corpus for reference
    corpus = [dictionary.doc2bow(text) for text in processed_clause]

    return corpus, dictionary


def train_lda_model(corpus, dictionary, topics):
    """
    Train the lda model
    :param corpus: corpus values
    :param dictionary: dictionary values
    :param topics: number of topics
    :return:
    """
    lda = models.LdaModel(corpus, num_topics=topics, id2word=dictionary, update_every=5, chunksize=10000, passes=100)

    return lda


def compute_coherence_values(dictionary, corpus, texts, limit, start=2):
    coherence_vals = []
    model_list = []
    for num_topics in range(start, limit):
        model = models.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)
        coherence_mdl = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_vals.append(coherence_mdl.get_coherence())

    return model_list, coherence_vals
