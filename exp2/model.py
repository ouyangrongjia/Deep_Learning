import csv
import numpy as np
import pandas as pd


def dataProcess_X(data):
    # 测试集中没有income行，单独判断 这两列都可以直接用一位二进制码表示 无需进行one-hot编码
    if 'income' in data.columns:
        Data = data.drop(['income', 'sex'], axis=1)
    else:
        Data = data.drop(['sex'], axis=1)

    # 提取离散属性列(如work-class,education等)
    listObjectData = [
        col for col in Data.columns if Data[col].dtypes == 'object'
    ]

    # 提取连续属性列(如age,capital_gain...) 直接使用数据中的原值
    listNonObjectData = [
        col for col in Data.columns if col not in listObjectData]

    ObjectData = Data[listObjectData]
    NonObjectData = Data[listNonObjectData]
    # print(NonObjectData)
    # 插入sex列，0代表male，1代表female
    # astype判断条件为真则插入1，否则插入0

    NonObjectData.insert(0, 'sex', (data['sex'] == ' Female').astype(np.int64))
    # 对于离散属性使用one-hot编码

    ObjectData = pd.get_dummies(ObjectData)

    Data = pd.concat([NonObjectData, ObjectData], axis=1)

    Data = Data.astype('int64')

    Data = (Data - Data.mean()) / Data.std()

    return Data


def dataProcess_Y(data):
    # income > 50K 设为1 否则设为0
    return (data["income"] == ' >50K').astype(np.int64)


def train(X, Y):
    train_data_size = X.shape[0]
    avg1 = np.zeros((106,))  # 类别1均值
    avg2 = np.zeros((106,))  # 类别2均值
    class1 = 0  # 类别1数量(1类别)
    class2 = 0  # 类别2数量(0类别)
    for i in range(train_data_size):
        if Y[i] == 1:  # 分类为 >50K 的类型1
            avg1 += X[i]
            class1 += 1
        else:  # 分类为 <=50K 的类型2
            avg2 += X[i]
            class2 += 1
    avg1 /= class1
    avg2 /= class2

    # 计算方差
    sigma1 = np.zeros((106, 106))  # 类别1方差
    sigma2 = np.zeros((106, 106))  # 类别2方差
    for i in range(train_data_size):
        if Y[i] == 1:
            sigma1 += np.dot(np.transpose([X[i] - avg1]), [X[i] - avg1])
        else:
            sigma2 += np.dot(np.transpose([X[i] - avg2]), [X[i] - avg2])

    sigma1 /= class1
    sigma2 /= class2
    # 计算协方差
    cov_sigma = (class1 / train_data_size) * sigma1 + (class2 / train_data_size) * sigma2

    return avg1, avg2, cov_sigma, class1, class2


def cal(test_x, avg1, avg2, cov_sigma, class1, class2):
    # 计算概率
    w = np.transpose(avg1 - avg2).dot(np.linalg.inv(cov_sigma))
    b = -0.5 * np.transpose(avg1).dot(np.linalg.inv(cov_sigma)).dot(avg1) + \
        0.5 * np.transpose(avg2).dot(np.linalg.inv(cov_sigma)).dot(avg2) + \
        np.log(float(class1 / class2))
    arr = np.empty([test_x.shape[0], 1], dtype=float)
    for i in range(test_x.shape[0]):
        # print(test_x[i, :])
        z = test_x[i, :].dot(w) + b
        z *= -1
        # print(z)
        arr[i][0] = 1 / (1 + np.exp(z))
    return np.clip(arr, 1e-8, 1 - 1e-8)


def predict(x):
    ans = np.zeros([x.shape[0], 1], dtype=int)
    # 对于获得的概率值，如果大于0.5，则认为其分类为1，否则认为其分类为0（收入<=50K)
    for i in range(x.shape[0]):
        if x[i] > 0.5:
            ans[i] = 1
        else:
            ans[i] = 0

    return ans


def writeFile(ans):
    with open('./data/predict.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(ans)):
            writer.writerow([i, ans[i]])

    file.close()


train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')
# 获取所有字符串列的列名
# string_columns = train_data.select_dtypes(include=['object']).columns
# # 去除每个字符串列的首尾空格
# train_data[string_columns] = train_data[string_columns].apply(lambda x: x.str.strip())
# 将数据首尾空格除去

# 对数据从107维降为106维，方便处理
X_train = dataProcess_X(train_data).drop(
    ['native_country_ Holand-Netherlands'], axis=1).values
Y_train = dataProcess_Y(train_data).values
X_test = dataProcess_X(test_data).values


# 计算概率所需的参数
mu1, mu2, shared_sigma, n1, n2 = train(X_train, Y_train)
result = cal(X_test, mu1, mu2, shared_sigma, n1, n2)
answer = predict(result)
print(answer[5:15])
writeFile(ans=answer)
