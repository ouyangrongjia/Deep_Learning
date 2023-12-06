from keras.models import Sequential
from keras.layers import Dense, Embedding, SimpleRNN, LSTM
from keras.datasets import imdb
from keras.preprocessing import sequence
import matplotlib.pyplot as plt

if __name__ == "__main__":
    max_features = 10000
    maxlen = 500
    batch_size = 32
    print("Loading data...")
    # 加载imdb代码，数据集地址为:/data/workspace/myshixun/imdb.npz
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

    print(len(X_train), "train sequences")
    print(len(X_test), "test sequences")

    print("Pad sequences (sample x times)")
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    model = Sequential()
    model.add(Embedding(max_features, 32))
    # 添加SimpleRNN层，参数为32
    model.add(SimpleRNN(32))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.savefig('./visual/acc.jpg')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.savefig('./visual/loss.jpg')
    plt.legend()
    plt.show()
