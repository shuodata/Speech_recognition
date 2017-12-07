__author__ = 'Joaquin Bejar Garcia: https://www.linkedin.com/in/joaquinbejar/'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import theano
import pygpu
import tensorflow as tf
import keras as k
import numpy as np
import scipy
import matplotlib as plt
import seaborn as sns
import IPython
import sklearn


if __name__ == "__main__":
    '''

    '''

    # Check Versions
    print('theano:', theano.__version__,' pygpu:', pygpu.__version__
    ,' tensorflow:', tf.__version__
    ,' Keras:', k.__version__
    ,' numpy:', np.__version__
    ,' scipy:', scipy.__version__
    ,' plt:', plt.__version__
    ,' seaborn:', sns.__version__
    ,' iPython:', IPython.__version__
    ,' scikit-learn:', sklearn.__version__)



    model = k.models.Sequential()


    model.add(k.layers.core.Dense(10, input_shape=(10,)))
    model.add(k.layers.core.Activation('sigmoid'))

    model.add(k.layers.core.Dense(10))
    model.add(k.layers.core.Activation('softmax'))

    model.add(k.layers.core.Dense(10, input_shape=(10,10,)))
    model.add(k.layers.core.Activation('relu'))




    model.compile(optimizer='adam', loss='mean_squared_error', metrics=["accuracy"])

    train_X = np.random.rand(10,10)
    test_X = np.random.rand(10,10)
    train_y = np.random.rand(10,10)
    test_y = np.random.rand(10,10)


    model.fit(train_X, train_y, epochs=100, batch_size=1, verbose=0);


    loss, accuracy = model.evaluate(test_X, test_y, verbose=0)

    output = model.predict(train_X, batch_size=1, verbose=0);
    print("Accuracy = {:.2f}".format(accuracy))

    print(output)
