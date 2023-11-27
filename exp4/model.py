import numpy as np
import keras.utils
from keras.datasets import mnist
from keras import Sequential
from keras.src.layers import Dense


def mnist_data():
    path = './data/mnist.npz'
    file = np.load(path)
    # download data from Google
    X_train, y_train, X_test, y_test = file['x_train'], file['y_train'], file['x_test'], file['y_test']
    # transfer the dtype of X_train and X_test
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    file.close()

    # 重塑数据集 转成(-1,784)
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # 归一化
    X_train = X_train / 255
    X_test = X_test / 255

    # one-hot编码
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    # build model
    # model = Sequential()
    # model.add(Dense(input_shape=(784,), units=10))
    # model.add(Activation('softmax'))
    # model.summary()
    # # compile network
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='SGD',
    #               metrics=['accuracy'])
    # # train network
    # # verbose=1 means output the details of log while training, validation_split=0,2 means using 20% of train_data as validation_data
    # model.fit(x=X_train, y=y_train, batch_size=128, epochs=200, verbose=1, validation_split=0.2)
    # # evaluate the model
    # score = model.evaluate(x=X_test, y=y_test, batch_size=32, verbose=1)
    # print("score loss: ", score[0])
    # print('score accuracy: ', score[1])

    # model optimization
    model2 = Sequential()
    # input layer
    model2.add(Dense(units=128, input_shape=(784,), activation='relu'))
    # hidden layer
    model2.add(Dense(units=128, activation='relu'), )
    # output layer
    model2.add(Dense(units=10, activation='softmax'))
    model2.summary()

    # compile network
    model2.compile(loss='categorical_crossentropy',
                   optimizer='SGD',
                   metrics=['accuracy'])

    # train network
    # verbose=1 means output the details of log while training, validation_split=0,2 means using 20% of train_data as validation_data
    model2.fit(x=X_train, y=y_train, batch_size=128, epochs=20, verbose=1, validation_split=0.2)

    # evaluate the model
    score = model2.evaluate(x=X_test, y=y_test, batch_size=32, verbose=1)
    print("Test score: ", score[0])
    print('Test accuracy: ', score[1])

    test_loss, test_acc = score[0], score[1]
    return test_loss, test_acc, model2

loss, acc, model = mnist_data()
model.save('./model.h5')
