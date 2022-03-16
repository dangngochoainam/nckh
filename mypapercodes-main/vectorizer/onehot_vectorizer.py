from keras.preprocessing.text import Tokenizer
from pandas.core.series import Series
import pandas as pd


def onehot_vectorizer(train_data=[], test_data=[], num_words=10000):
    if type(train_data) == Series and type(test_data) == Series:
        data = pd.concat([train_data, test_data])
    else:
        data = train_data + test_data

    # tokenizer = Tokenizer(num_words=num_words)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)

    results = tokenizer.texts_to_matrix(data, mode='binary')

    # results = tokenizer.texts_to_matrix(data, mode='count')
    # ‘binary‘: Whether or not each word is present in the document. This is the default.
    # 'count‘: The count of each word in the document.
    # 'tfidf‘: The Text Frequency - Inverse DocumentFrequency(TF - IDF) scoring for each word in the document.
    # 'freq': The frequency of each word as a ratio of words within each document.

    # print(results)

    # input_shape = (data.shape[0], len(tokenizer.word_index))
    idx = len(train_data)
    return results[:idx], results[idx:]


if __name__ == '__main__':
    data = Series(["This is a book", "This book is good"])
    test = Series(["I love read a book"])
    onehot_vectorizer(data, test, num_words=8)
