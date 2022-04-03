import utils
import pandas as pd
from utils.read_data import read_news_dataset, save_cached_file, load_cached_file
from vectorizer.tfidf_vectorizer import tfidf_vectorizer
from classifiers.classifiers import (logistic_regression_train,
                                     svm_train,
                                     multi_nomial_naive_bayes_train,
                                     base_prediction)
from sklearn.model_selection import train_test_split
from neural_networks.dnn import dnn
from pprint import pprint
import numpy as np
from preprocess.preprocess import execute_new_data
from utils import BASE_DIR, generate_data
import os

def execute_dnn(cached_file=None, cached_file_aug=None):
    data = read_news_dataset(is_preprocess=True,
                             is_forced=False,
                             cached_file=cached_file)


    train_data, test_data, train_targets, test_targets \
        = train_test_split(data.feature, data.target,
                           test_size=0.3,
                           # test_size=0.999,
                           random_state=42)

    # if cached_file_aug:
    #     cached_path = os.path.join(BASE_DIR, cached_file_aug)
    #     new_data = load_cached_file(cached_file_path=cached_path)
    # else:
    #     new_data = generate_data(train_data, train_targets)
    #     save_cached_file(new_data, '%s/cached_data/news_dataset.sav' %BASE_DIR)
    #
    # df = pd.DataFrame(new_data)
    #
    # train_data = train_data.append(df.feature)
    # train_targets = train_targets.append(df.target)


    train_vectors, test_vectors = tfidf_vectorizer(train_data=train_data, test_data=test_data)
    dnn(train_vectors=train_vectors.toarray(), train_target=train_targets,
        test_vectors=test_vectors.toarray(), test_target=test_targets)


    print(np.shape(train_vectors))


def execute_classifiers(classifier_method, cached_file=None, cached_file_aug=None):


    data = read_news_dataset(is_preprocess=True,
                             is_forced=False,
                             cached_file=cached_file)


    train_data, test_data, train_targets, test_targets \
        = train_test_split(data.feature, data.target,
                           # test_size=0.999,
                           test_size=0.3,
                           random_state=42)


    # if cached_file_aug:
    #     cached_path = os.path.join(BASE_DIR, cached_file_aug)
    #     new_data = load_cached_file(cached_file_path=cached_path)
    # else:
    #     new_data = generate_data(train_data, train_targets)
    #     save_cached_file(new_data, '%s/cached_data/news_dataset.sav' %BASE_DIR)
    #
    # df = pd.DataFrame(new_data)
    #
    # train_data = train_data.append(df.feature)
    # train_targets = train_targets.append(df.target)


    train_vectors, test_vectors = tfidf_vectorizer(train_data=train_data,
                                                   test_data=test_data)
    print(np.shape(train_vectors))

    cls = classifier_method(train_vectors=train_vectors, train_targets=train_targets)

    results = base_prediction(classifier=cls,
                              test_vectors=test_vectors,
                              test_target=test_targets)

    pprint(results)


def execute_logistic_regression(cached_file=None, cached_file_aug=None):
    execute_classifiers(classifier_method=logistic_regression_train, cached_file=cached_file, cached_file_aug=cached_file_aug)


def execute_svm():
    execute_classifiers(classifier_method=svm_train)


def execute_multi_nomial_nb():
    execute_classifiers(classifier_method=multi_nomial_naive_bayes_train)


if __name__ == '__main__':
    # execute_dnn(cached_file='cached_data/dataset_temp15.sav', cached_file_aug=None)
    execute_logistic_regression(cached_file='cached_data/dataset_temp15.sav', cached_file_aug=None)


