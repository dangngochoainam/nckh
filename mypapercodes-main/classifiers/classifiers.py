from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


def base_classifier(classifier, params, train_vectors, train_targets):
    cls = classifier(**params)
    cls.fit(train_vectors, train_targets)

    return cls


def base_prediction(classifier, test_vectors, test_target):
    predict = classifier.predict(test_vectors)

    results = {
        'accuracy': metrics.accuracy_score(test_target, predict),
        'f1_score': metrics.f1_score(test_target, predict, average='macro')
    }

    return results


def logistic_regression_train(train_vectors, train_targets, custom_params=None):
    params = {
        'multi_class': 'multinomial',# ovr
        'solver': 'lbfgs'
    } if not custom_params else custom_params

    return base_classifier(classifier=LogisticRegression,
                           params=params,
                           train_vectors=train_vectors,
                           train_targets=train_targets)


def svm_train(train_vectors, train_targets, custom_params=None):
    params = {
        'kernel': 'rbf',
        'gamma': 'auto',
        'degree': 3,
        'coef0': 0,
        'C': 1e5
    } if not custom_params else custom_params

    return base_classifier(classifier=SVC,
                           params=params,
                           train_vectors=train_vectors,
                           train_targets=train_targets)


def multi_nomial_naive_bayes_train(train_vectors, train_targets, custom_params=None):
    params = {
        'alpha': 0.01
    } if not custom_params else custom_params

    return base_classifier(classifier=MultinomialNB,
                           params=params,
                           train_vectors=train_vectors,
                           train_targets=train_targets)


def bernoulli_naive_bayes_train(train_vectors, train_target, custom_params=None):
    params = {
        'alpha': 0.01
    } if not custom_params else custom_params

    return base_classifier(classifier=BernoulliNB,
                           params=params,
                           train_vectors=train_vectors,
                           train_targets=train_target)


def knn_train(train_vectors, train_targets, custom_vectors=None):
    params = {
        'n_neighbors': 1,
        'p': 2
    } if not custom_vectors else custom_vectors

    return base_classifier(classifier=KNeighborsClassifier,
                           params=params,
                           train_vectors=train_vectors,
                           train_targets=train_targets)
