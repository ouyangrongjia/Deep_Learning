import math

import pandas as pd
import csv
import numpy as np

# 第一步：读取数据，GBK编码方法针对数据中存在的繁体字
train_data = []
for i in range(18):
    train_data.append([])
n_row = 0
text = open('data/train.csv', 'r', errors='ignore', encoding='GBK')
row = csv.reader(text, delimiter=',')
for r in row:
    # 每一行只有第3-27格有值(即一天中24小时)
    if n_row > 0:
        for i in range(3, 27):
            if r[i] != "NR":  # 其中有一个污染物全部值为‘NR’
                train_data[(n_row - 1) % 18].append(float(r[i]))
            else:
                train_data[(n_row - 1) % 18].append(float(0))
    n_row = n_row + 1
text.close()
# for i in range(18):
#     print(train_data[i])


x = []  # 特征
y = []  # 标签
# 共有12个月
for i in range(12):
    # 每个月共有480列数据，连取10小时的分组可有471组。
    for j in range(471):
        x.append([])
        # 共有18种污染物
        for t in range(18):  # 把18行合成同一行
            # 取前9小时为feature
            for s in range(9):
                x[471 * i + j].append(train_data[t][480 * i + j + s])
        y.append(train_data[9][480 * i + j + 9])  # 取PM2.5的标签
x = np.array(x)
y = np.array(y)
# 在第一列添加一列1
x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

#线性回归
weight = np.zeros(len(x[0]))
learning_rate = 10
epoch = 100
x_t = x.transpose()  # x转置
s_gradient = np.zeros(len(x[0]))

for i in range(epoch):  # 每一次迭代
    hypo = np.dot(x, weight)  # weight*x
    loss = hypo - y

    cost = np.sum(loss**2) / len(x)  # 平方差
    cost_a = math.sqrt(cost)  # 标准差
    gra = np.dot(x_t, loss)  # 可x2

    s_gradient += gra**2
    ada = np.sqrt(s_gradient)
    weight = weight - learning_rate * gra/ada

# 保存模型
np.save('model.npy', weight)
# 使用刚刚训练得到的模型进行预测
weight = np.load('model.npy')

text = open('data/test.csv', 'r')
row = csv.reader(text, delimiter=',')
n_row = 0
x_test = []
for r in row:
    if n_row % 18 == 0:
        x_test.append([])  #  每18个加一行
        for i in range(2, 11):
            x_test[n_row//18].append(float(r[i]))  #  整除18，得到每次id的预测
    else:
        for i in range(2, 11):
            if r[i] != 'NR':
                x_test[n_row//18].append(float(r[i]))
            else:
                x_test[n_row//18].append(float(0))
    n_row = n_row + 1
text.close()
x_test = np.array(x_test)
x_test = np.concatenate((np.ones((x_test.shape[0], 1)), x_test), axis=1)

# for i in range(240):
#     print(x_test[i])
# ans = []
# for i in range(len(x_test)):
#     ans.append(["id_"+str(i)])
#     a = np.dot(weight, x_test[i])  # 使用训练好的权重来完成测试集的预测
#     ans[i].append(a)
# #######End#######
# # 打印预测结果的信息
# print("共有预测结果%d条"%(len(ans)))

pre_list = []
for i in range(len(x_test)):
    pre = weight.dot(x_test[i])
    pre_list.append(pre)
    print('id_', i, pre)

# 6.3 保存预测结果到文件中
# 打开文件，准备写入
newfile = open('data/sampleSubmission.csv', 'w', newline='')
writer = csv.writer(newfile, delimiter=',', lineterminator='\n')
for i in range(len(x_test)):
    writer.writerow(['id_'+str(i), pre_list[i]])
newfile.close()



