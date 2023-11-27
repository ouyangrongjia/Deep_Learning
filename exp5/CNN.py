# 1.1. 导包
import numpy as np
import math
import matplotlib.pyplot as plt
import keras.utils
from keras import models
from keras.layers import InputLayer, Input, Reshape, MaxPooling2D, Conv2D, Dense, Flatten
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import backend

# 1.2. 载入数据
(X_train, y_train), (X_test, y_test) = mnist.load_data()
img_size = 28  # 图像维度
img_size_flat = 28 * 28  # 将图像重塑为一维的长度
img_shape = (28, 28)  # 重塑图像的高度和宽度的元组
img_shape_full = (28, 28, 1)  # 重塑图像的高度，宽度和深度的元组
num_classes = 10  # 类别数量
num_channels = 1  # 通道数
path_model = './model2.pkl'

# 1.3．配置神经网络
# 请打印并查看上述定义和赋值的变量。
print(img_size)
print(img_size_flat)
print(img_shape)
print(img_shape_full)
print(num_classes)
print(num_channels)

# 1.4.绘制图像的辅助函数
def plot_images(images, cls_true, id, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)  # 分为3*3个子图
    fig.subplots_adjust(hspace=0.3, wspace=0.3)  # 每个子图之间的间距

    for (i, ax) in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap="binary")
        if cls_pred is None:
            xlabel = "True:{0}".format(cls_true[i])
        else:
            xlabel = "True:{0}, Pred:{1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)
        # 去除刻度线
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig('./visual/predict' + id + '.jpg')
    plt.show()

# 1.5.绘制错误分类图像的辅助函数
def plot_example_errors(cls_pred, correct, id):
    incorrect = (correct == False)
    images = X_test[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = y_test[incorrect]
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9], id=id, cls_pred=cls_pred[0:9])

# 1.23 权重和输出的可视化函数
def plot_conv_weights(weights, id, input_channel=0):
    #获取权重的最低值和最高值
    # 这用于校正图像的颜色强度，以便可以相互比较.
    w_min = np.min(weights)
    w_max = np.max(weights)
    # 卷积层中的卷积核数量
    num_filters = weights.shape[3]
    #要绘制的网格数.
    # 卷积核的平方根.
    num_grids = math.ceil(math.sqrt(num_filters))
    #创建带有网格子图的图像.
    fig, axes = plt.subplots(num_grids, num_grids)
    #  绘制所有卷积核的权重
    for i, ax in enumerate(axes.flat):
        #  仅绘制有限的卷积核权重
        if i<num_filters:
            #  获取输入通道的第i个卷积核的权重
            #   有关于４维张量格式的详细信息请参阅new_conv_layer()
            img = weights[:, :, input_channel, i]
            # 画图
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        # 去除刻度线.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig('./visual/predict' + id + '.jpg')
    plt.show()

# 1.26 绘制出卷积层输出的帮助函数
def plot_conv_output(values, id):
    #  卷积层中卷积核数量
    num_filters = values.shape[3]
    # 要绘制的网格数
    num_grids = math.ceil(math.sqrt(num_filters))
    # 创建子图
    fig, axes = plt.subplots(num_grids, num_grids)
    # 画出所有卷积核输出图像
    for (i, ax) in enumerate(axes.flat):
        # 仅画出有效卷积核图像
        if i < num_filters:
            # 获取输入通道第i个卷积核的权重
            img = values[0, :, :, i]
            # 画图
            ax.imshow(img, interpolation='nearest', cmap='binary')
        # 去除刻度线
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig('./visual/predict' + id + '.jpg')
    plt.show()

# 1.27 输入图像的展示函数
def plot_image(image):
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest', cmap='binary')
    plt.show()
# 1.6. 实现序列模型
# 序列模型
model = models.Sequential()
# 输入层
model.add(InputLayer(input_shape=(img_size_flat,)))
# 输入是一个包含784个元素的数组
# 但卷积层的期望图像是(28, 28, 1)
model.add(Reshape(img_shape_full))
# 具有ReLu激活和最大池化的卷积层1
model.add(Conv2D(kernel_size=5, strides=1, filters=16,
                 padding='same', activation='relu', name='layer_conv1'))
model.add(MaxPooling2D(pool_size=2, strides=2))
# 具有ReLu激活和最大池化的卷积层2
model.add(Conv2D(kernel_size=5, strides=1, filters=36,
                 padding='same', activation='relu', name='layer_conv2'))
model.add(MaxPooling2D(pool_size=2, strides=2))
# 将卷积层的输出拉直
model.add(Flatten())
# 具有ReLu激活的完全连接层
model.add(Dense(128, activation='relu'))
# 最后一个全连接层，具有softmax激活，用于分类
model.add(Dense(num_classes, activation='softmax'))
model.summary()

# 1.7. 编译模型
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='categorical_crossentropy', metrics=['accuracy'])

# 1.8. 将图像摊平为向量
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
# 归一化
X_train = X_train / 255
X_test = X_test / 255
y_train = keras.utils.to_categorical(y_train, 10)
y_test_h = keras.utils.to_categorical(y_test, 10)

# 1.9 训练模型
model.fit(X_train, y_train, epochs=1, batch_size=128,
          validation_split=1/12, verbose=2)

# 1.10 评估模型
result = model.evaluate(X_test, y_test_h, verbose=1)
print("loss ", result[0])
print("acc ", result[1])

# 1.11 用模型来预测
predict = model.predict(X_test)
print("1 predict: ", predict)
predict = np.argmax(predict, axis=1)
print("2 predict: ", predict)
plot_images(X_test[0:9], y_test[0:9], '1', predict[0:9])

# 1.12 错分类的图片
y_pred = model.predict(X_test)
cls_pred = np.argmax(y_pred, axis=1)
correct = (cls_pred == y_test)
plot_example_errors(cls_pred, correct=correct, id='2')

# 1.13 改进模型
# 输入层
inputs = Input(shape=(img_size_flat,))
# 构建神经网络变量
net = inputs
# 输入是一个包含784个元素的数组
# 但卷积层的期望图像是(28, 28, 1)
net = Reshape(img_shape_full)(net)
# 具有ReLu激活和最大池化的卷积层1
net = Conv2D(kernel_size=5, strides=1, filters=16,
             padding='same', activation='relu', name='layer_conv1')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)
# 具有ReLu激活和最大池化的卷积层1
net = Conv2D(kernel_size=5, strides=1, filters=36,
             padding='same', activation='relu', name='layer_conv2')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)
# 将卷积层输出拉直
net = Flatten()(net)
# 具有ReLu激活的完全连接层
net = Dense(128, activation='relu')(net)
# 最后一个全连接层，具有softmax激活，用于分类
net = Dense(num_classes, activation='softmax')(net)
# 输出
outputs = net

# 1.14 模型编译
model2 = models.Model(inputs=inputs, outputs=outputs)
model2.compile(optimizer='rmsprop',
               loss='categorical_crossentropy', metrics=['accuracy'])

# 1.15 训练模型
model2.fit(X_train, y_train, batch_size=128,
           epochs=1, validation_split=1/12, verbose=2)

# 1.16 评估模型
result = model2.evaluate(X_test, y_test_h, verbose=1)
print(model2.metrics_names[0], result[0])
print(model2.metrics_names[1], result[1])

# 1.17 预测
predict = model2.predict(X_test)
print("1 predict: ", predict)
predict = np.argmax(predict, axis=1)
print("2 predict: ", predict)
plot_images(X_test[0:9], y_test[0:9], '3', predict[0:9])

# 1.18 错分类图片
y_pred = model2.predict(X_test)
cls_pred = np.argmax(y_pred, axis=1)
correct = (cls_pred == y_test)
plot_example_errors(cls_pred, correct=correct, id='4')

# 1.19 保存模型
model2.save(path_model)

# 1.20 删除模型
del model2

# 1.21 加载模型3
model3 = models.load_model(path_model)

# 1.22 模型3预测
predict = model3.predict(X_test)
predict = np.argmax(predict, axis=1)
plot_images(X_test[0:9], y_test[0:9], '5', predict[0:9])

# 1.24 得到层
model3.summary()
layer_input = model3.layers[0]
layer_conv1 = model3.layers[2]
layer_conv2 = model3.layers[4]

# 1.25 卷积权重
weights_conv1 = layer_conv1.get_weights()[0]
plot_conv_weights(weights_conv1, '6', input_channel=0)
weights_conv2 = layer_conv2.get_weights()[0]
plot_conv_weights(weights_conv2, '7', input_channel=0)

# 1.27 输入图像
image1 = X_test[0]
plot_image(image1)

# 1.28 卷积层输出一
output_conv1 = backend.function(
        inputs=[layer_input.input], outputs=[layer_conv1.output])

# 1.29 获取卷积层输出一
layer_output1 = output_conv1([np.array([image1])])[0]
print(layer_output1.shape)

# 1.30 绘制输出
plot_conv_output(layer_output1, '8')

# 1.31 卷积层输出二
output_conv2 = backend.function(
    inputs=[layer_input.input], outputs=[layer_conv2.output])

# 1.32 获取卷积层输出二
layer_output2 = output_conv2([np.array([image1])])[0]
print(layer_output2.shape)

# 1.33 绘制输出
plot_conv_output(values=layer_output2, id='9')

