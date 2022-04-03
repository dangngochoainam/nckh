from keras import models, layers, Input, regularizers
from keras.layers import Dense
from keras.models import Model, Sequential
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from keras.models import load_model
from keras.optimizers import gradient_descent_v2
from keras.utils.np_utils import to_categorical







def dnn(train_vectors, train_target, test_vectors, test_target):

    model = models.Sequential()

    # model.add(layers.Dense(128, activation='relu', input_shape=(train_vectors.shape[1],)))
    # model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dense(32, activation='relu'))

    model.add(layers.Dense(128, activation='relu', input_shape=(train_vectors.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))

    # model.add(layers.Dense(25, activation='softmax'))
    model.add(layers.Dense(25, activation='sigmoid'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


    model.fit(train_vectors, train_target,
              epochs=20, batch_size=32,
              validation_data=(test_vectors, test_target))

    # print(model.summary())

    results = model.evaluate(test_vectors, test_target,
                             batch_size=512, verbose=False)
    print(results)

    # model.save('encoder.h5')

    # encoder = load_model('encoder.h5')
    # # encode the train data
    # X_train_encode = encoder.predict(train_vectors)
    # # encode the test data
    # X_test_encode = encoder.predict(test_vectors)
    # # define the model
    # model = LogisticRegression()
    # # fit the model on the training set
    # model.fit(X_train_encode, train_target)
    # # make predictions on the test set
    # yhat = model.predict(X_test_encode)
    # # calculate classification accuracy
    # acc = accuracy_score(test_target, yhat)
    # print(acc)




