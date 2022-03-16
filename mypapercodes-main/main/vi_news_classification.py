import utils
from utils.read_data import read_news_dataset
from vectorizer.tfidf_vectorizer import tfidf_vectorizer
from vectorizer.onehot_vectorizer import onehot_vectorizer
from classifiers.classifiers import (logistic_regression_train,
                                     svm_train,
                                     multi_nomial_naive_bayes_train,
                                     base_prediction)
from sklearn.model_selection import train_test_split
from neural_networks.dnn import dnn
from pprint import pprint

import numpy as np
# from neural_networks.test import dnn

def execute_dnn(path=None):
    data = read_news_dataset(path=path,
                             is_preprocess=False,
                             is_forced=True) if path else read_news_dataset()


    train_data, test_data, train_targets, test_targets \
        = train_test_split(data.feature, data.target,
                           # test_size=0.3,
                           test_size=0.99,
                           random_state=42)


    train_vectors, test_vectors = tfidf_vectorizer(train_data=train_data, test_data=test_data)
    dnn(train_vectors=train_vectors.toarray(), train_target=train_targets,
        test_vectors=test_vectors.toarray(), test_target=test_targets)



    # train_vectors, test_vectors = onehot_vectorizer(train_data=train_data, test_data=test_data)
    # dnn(train_vectors=train_vectors, train_target=train_targets,
    #     test_vectors=test_vectors, test_target=test_targets)

    print(np.shape(train_vectors))
    # print(train_vectors.toarray())


def execute_classifiers(classifier_method, path=None):
    data = read_news_dataset(path=path,
                             is_preprocess=True,
                             is_forced=True) if path else read_news_dataset()

    train_data, test_data, train_targets, test_targets \
        = train_test_split(data.feature, data.target,
                           test_size=0.99,
                           # test_size=0.3,
                           random_state=42)

    train_vectors, test_vectors = tfidf_vectorizer(train_data=train_data,
                                                   test_data=test_data)
    print(np.shape(train_vectors))

    cls = classifier_method(train_vectors=train_vectors, train_targets=train_targets)

    results = base_prediction(classifier=cls,
                              test_vectors=test_vectors,
                              test_target=test_targets)

    pprint(results)


def execute_logistic_regression(path=None):
    execute_classifiers(classifier_method=logistic_regression_train,
                        path=path)


def execute_svm(path=None):
    execute_classifiers(classifier_method=svm_train,
                        path=path)


def execute_multi_nomial_nb(path=None):
    execute_classifiers(classifier_method=multi_nomial_naive_bayes_train,
                        path=path)


if __name__ == '__main__':
    # execute_dnn(path=utils.sentiment1_dataset_path)
    execute_dnn()
    # execute_multi_nomial_nb()
    # execute_logistic_regression(path=utils.sentiment1_dataset_path)
    # execute_logistic_regression()
    # execute_svm()


