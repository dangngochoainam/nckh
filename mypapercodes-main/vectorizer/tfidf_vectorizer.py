from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_vectorizer(train_data, test_data, custom_params={}):
    # 'ngram_range': (1, 2)
    v = TfidfVectorizer(**custom_params)

    train_vectors = v.fit_transform(train_data)
    test_vectors = v.transform(test_data)

    return train_vectors, test_vectors
