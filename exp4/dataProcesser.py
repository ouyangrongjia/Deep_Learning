import numpy as np
from keras.datasets import mnist
import keras.utils

# (X_train, y_train), (X_test, y_test) = mnist.load_data()
path = './data/mnist.npz'  # mnist数据集的文件路径

f = np.load(path)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
f.close()

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# 重塑数据集 转成(-1,784)

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print(X_train.shape)
print(X_test.shape)
print(X_train.dtype)
print(X_test.dtype)

# 归一化

X_train = X_train/255
X_test = X_test/255
print(X_train[1][100:150])

# one-hot编码

y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

print(y_train[:10])
print(y_test[:10])
