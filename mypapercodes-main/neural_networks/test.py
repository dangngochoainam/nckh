from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.utils.vis_utils import plot_model
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from keras.models import load_model
from utils.read_data import read_news_dataset
from vectorizer.tfidf_vectorizer import tfidf_vectorizer





# from keras import models, layers, Input, regularizers
# from keras.layers import Dense
# from keras.models import Model, Sequential
# import numpy as np
#
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
#
#
#
#
#
# def dnn(train_vectors, train_target, test_vectors, test_target):
#     model = models.Sequential()
#
#     model.add(layers.Dense(32, activation='relu', input_shape=(30160, )))
#     model.add(layers.Dense(16, activation='relu'))
#     model.add(layers.Dense(16, activation='relu'))
#
#     model.add(layers.Dense(1, activation='sigmoid'))
#
#
#     model.compile(optimizer='adam', loss='mse', #rmsprop
#                   metrics=['accuracy'])
#
#
#
#     # chính
#     model.fit(train_vectors, train_target,
#               epochs=10, batch_size=512,
#               validation_data=(test_vectors, test_target))
#
#
#
#
#
#
#     print(model.summary())
#
#     # chính
#     results = model.evaluate(test_vectors, test_target,
#                              batch_size=512, verbose=True)
#     print(results)
#






if __name__ == '__main__':



    # # define dataset
    path = None
    data = read_news_dataset(path=path,
                             is_preprocess=False,
                             is_forced=True) if path else read_news_dataset()
    # X, y = make_classification(n_samples=1000, n_features=100, n_informative=10, n_redundant=90, random_state=1, n_classes=3, n_clusters_per_class=3)

    # split into train test sets
    X_train, X_test, y_train, y_test = train_test_split(data.feature, data.target, test_size=0.99, random_state=1)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


    # scale data
    X_train, X_test = tfidf_vectorizer(train_data=X_train,
                                       test_data=X_test)
    X_train = X_train.toarray()
    X_test = X_test.toarray()

    # number of input columns
    n_inputs = X_train.shape[1]

    # # t = MinMaxScaler()
    # # t.fit(X_train)
    # # X_train = t.transform(X_train)
    # # X_test = t.transform(X_test)
    #
    # define encoder
    visible = Input(shape=(n_inputs,))
    # encoder level 1
    e = Dense(n_inputs*2)(visible)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)
    # encoder level 2
    e = Dense(n_inputs)(e)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)
    # bottleneck
    n_bottleneck = round(float(n_inputs) / 2.0)
    bottleneck = Dense(n_bottleneck)(e)
    # define decoder, level 1
    d = Dense(n_inputs)(bottleneck)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)
    # decoder level 2
    d = Dense(n_inputs*2)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)
    # output layer
    # output = Dense(n_inputs, activation='linear')(d)
    output = Dense(n_inputs, activation='softmax')(d)

    # define autoencoder model
    model = Model(inputs=visible, outputs=output)
    # compile autoencoder model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    # plot the autoencoder
    # plot_model(model, 'autoencoder_no_compress.png', show_shapes=True)
    # fit the autoencoder model to reconstruct input
    history = model.fit(X_train, X_train, epochs=20, batch_size=16, validation_data=(X_test, X_test))
    # plot loss
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    # define an encoder model (without the decoder)
    encoder = Model(inputs=visible, outputs=bottleneck)
    # plot_model(encoder, 'encoder_no_compress.png', show_shapes=True)
    # save the encoder to file
    encoder.save('encoder.h5')


    # path = None
    # data = read_news_dataset(path=path,
    #                          is_preprocess=False,
    #                          is_forced=True) if path else read_news_dataset()
    #
    # X_train, X_test, y_train, y_test = train_test_split(data.feature, data.target, test_size=0.33, random_state=1)
    # # scale data
    # X_train, X_test = tfidf_vectorizer(train_data=X_train,
    #                                                test_data=X_test)
    # X_train = X_train.toarray()
    # X_test = X_test.toarray()
    # # t = MinMaxScaler()
    # # t.fit(X_train)
    # # X_train = t.transform(X_train)
    # # X_test = t.transform(X_test)
    # # load the model from file
    # encoder = load_model('encoder.h5')
    # # encode the train data
    # X_train_encode = encoder.predict(X_train)
    # # encode the test data
    # X_test_encode = encoder.predict(X_test)
    # # define the model
    # model = LogisticRegression()
    # # fit the model on the training set
    # model.fit(X_train_encode, y_train)
    # # make predictions on the test set
    # yhat = model.predict(X_test_encode)
    # # calculate classification accuracy
    # acc = accuracy_score(y_test, yhat)
    # print(acc)




    # # split into train test sets
    # path = None
    # data = read_news_dataset(path=path,
    #                          is_preprocess=False,
    #                          is_forced=True) if path else read_news_dataset()
    #
    # X_train, X_test, y_train, y_test = train_test_split(data.feature, data.target, test_size=0.33, random_state=1)
    # # scale data
    # # print(X_test)
    # # print('====')
    #
    # # t = MinMaxScaler()
    # # t.fit(X_train)
    # # X_train = t.transform(X_train)
    # # X_test = t.transform(X_test)
    # # print(X_train)
    # X_train, X_test = tfidf_vectorizer(train_data=X_train,
    #                                                test_data=X_test)
    # # define model
    # model = LogisticRegression()
    # # fit model on training set
    # model.fit(X_train, y_train)
    # # make prediction on test set
    # yhat = model.predict(X_test)
    # # calculate accuracy
    # acc = accuracy_score(y_test, yhat)
    # print(acc)

