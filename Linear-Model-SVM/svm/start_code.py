from numpy.core.fromnumeric import shape
import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import matplotlib.pyplot as plt
import re


def load_text_dataset(filename, positive='joy', negative='sadness'):
    """
    从文件filename读入文本数据集
    """
    data = pd.read_csv('D:/SomeCodes/ml-assignment1/svm/' + filename)
    is_positive = data.Emotion == positive
    is_negative = data.Emotion == negative
    data = data[is_positive | is_negative]
    X = data.Text  # 输入文本
    y = np.array(data.Emotion == positive) * 2 - 1  # 1: positive, -1: negative
    return X, y


def preprocess_and_tokenize(data):
    """文本的分词"""
    # remove html markup
    data = re.sub("(<.*?>)", "", data)

    # remove urls
    data = re.sub(r'http\S+', '', data)

    # remove hashtags and @names
    data = re.sub(r"(#[\d\w\.]+)", '', data)
    data = re.sub(r"(@[\d\w\.]+)", '', data)

    # remove punctuation and non-ascii digits
    data = re.sub("(\\W|\\d)", " ", data)

    # remove whitespace
    data = data.strip()

    # tokenization with nltk
    data = word_tokenize(data)

    # stemming with nltk
    porter = PorterStemmer()
    stem_data = [porter.stem(word) for word in data]

    return stem_data


def vectorize(train, test):
    """
    将训练集和验证集中的文本转成向量表示

    Args：
        train - 训练集，大小为 num_instances 的文本数组
        test - 测试集，大小为 num_instances 的文本数组
    Return：
        train_normalized - 向量化的训练集
        test_normalized - 向量化的测试集
    """
    # TODO
    vectorizer = TfidfVectorizer()
    train_normalized = vectorizer.fit_transform(train)
    test_normalized = vectorizer.transform(test)
    return train_normalized.toarray(), test_normalized.toarray()


def compute_accuracy(X, y, theta):
    y_pred = np.where(X.dot(theta) >= 0, 1, -1)
    return np.count_nonzero(y == y_pred) / X.shape[0]


def linear_svm_subgrad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000, batch_size=1):
    """
    线性SVM的随机次梯度下降

    参数：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        alpha - 浮点数。初始梯度下降步长
        lambda_reg - 正则化系数
        num_iter - 遍历整个训练集的次数（即次数）
        batch_size - 批大小

    返回：
        theta_hist - 参数向量的历史，大小的 3D numpy 数组 (num_iter, num_instances, num_features)
        loss hist - 正则化损失函数向量的历史，数组大小(num_iter, num_instances)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features)  # Initialize theta
    theta_hist = np.zeros((num_iter+1, num_features))  # Initialize theta_hist
    loss_hist = np.zeros(num_iter+1)  # Initialize loss_hist

    # TODO
    t = 0
    theta_hist[0] = theta
    loss_hist[0] = np.average(np.maximum(0, 1 - y * X.dot(theta)))
    eta = alpha

    for iter in range(1, num_iter + 1):
        t += 1
        eta = 1 / (lambda_reg * t)

        sample_index = np.random.randint(num_instances, size=batch_size)
        X_sample, y_sample = X[sample_index], y[sample_index]
        loss_gradient = np.zeros(shape=(batch_size, num_features))
        for i in range(batch_size):
            if y_sample[i] * (X_sample[i].dot(theta)) < 1:
                loss_gradient[i] = lambda_reg * theta - y_sample[i] * X_sample[i]
            else:
                loss_gradient[i] = lambda_reg * theta
        loss_gradient = np.average(loss_gradient, axis=0)
        theta = theta - eta * loss_gradient
        
        if iter % 1 == 0:
            print("iter {}: accuracy = {}".format(iter, compute_accuracy(X, y, theta)))
        
        theta_hist[iter] = theta
        loss_hist[iter] = np.average(np.maximum(0, 1 - y * X.dot(theta)))

    return theta_hist, loss_hist


def get_mapping_matrix(X):
    m, n = X.shape
    return np.hstack((np.eye(n), 0.01 * np.tril(np.ones((n, 1)))))


def kernel_svm_subgrad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000, batch_size=1):
    """
    Kernel SVM的随机次梯度下降

    参数：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        alpha - 浮点数。初始梯度下降步长
        lambda_reg - 正则化系数
        num_iter - 遍历整个训练集的次数（即次数）
        batch_size - 批大小

    返回：
        theta_hist - 参数向量的历史，大小的 3D numpy 数组 (num_iter, num_instances, num_features)
        loss hist - 正则化损失函数向量的历史，数组大小(num_iter, num_instances)
    """
    X = X.dot(get_mapping_matrix(X))
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features)  # Initialize theta
    theta_hist = np.zeros((num_iter+1, num_features))  # Initialize theta_hist
    loss_hist = np.zeros(num_iter+1)  # Initialize loss_hist

    # TODO
    t = 0
    theta_hist[0] = theta
    loss_hist[0] = np.average(np.maximum(0, 1 - y * X.dot(theta)))
    eta = alpha

    for iter in range(1, num_iter + 1):
        t += 1
        eta = 1 / (lambda_reg * t)

        sample_index = np.random.randint(num_instances, size=batch_size)
        X_sample, y_sample = X[sample_index], y[sample_index]
        loss_gradient = np.zeros(shape=(batch_size, num_features))
        for i in range(batch_size):
            if y_sample[i] * (X_sample[i].dot(theta)) < 1:
                loss_gradient[i] = lambda_reg * theta - y_sample[i] * X_sample[i]
            else:
                loss_gradient[i] = lambda_reg * theta
        loss_gradient = np.average(loss_gradient, axis=0)
        theta = theta - eta * loss_gradient
        
        if iter % 1 == 0:
            print("iter {}: accuracy = {}".format(iter, compute_accuracy(X, y, theta)))
        
        theta_hist[iter] = theta
        loss_hist[iter] = np.average(np.maximum(0, 1 - y * X.dot(theta)))

    return theta_hist, loss_hist

def main():
    # 加载所有数据
    X_train, y_train = load_text_dataset("data_train.csv", "joy", "sadness")
    X_val, y_val = load_text_dataset("data_test.csv")
    print("Training Set Size: {} Validation Set Size: {}".format(len(X_train), len(X_val)))
    
    # 将训练集和验证集中的文本转成向量表示
    X_train_vect, X_val_vect = vectorize(X_train, X_val)

    # SVM的次梯度下降训练
    # TODO
    # theta_hist, loss_hist = linear_svm_subgrad_descent(X_train_vect, y_train, lambda_reg=1e-4, num_iter=200, batch_size=512)

    # # 计算SVM模型在验证集上的准确率，F1-Score以及混淆矩阵
    # # TODO
    # print("================= Test Result =================")
    # print(compute_accuracy(X_val_vect, y_val, theta_hist[-1]))
    # y_pred = np.where(X_val_vect.dot(theta_hist[-1]) >= 0, 1, -1)
    # print("Accuracy:%.4f" % (metrics.accuracy_score(y_val, y_pred)))
    # print("F1-score:%.4f" % (metrics.f1_score(y_val, y_pred)))
    # confusion_matrix = metrics.confusion_matrix(y_val, y_pred)
    # indices = range(len(confusion_matrix))
    # plt.imshow(confusion_matrix, cmap=plt.cm.Blues)
    # plt.xticks(indices, [-1, 1])
    # plt.yticks(indices, [-1, 1])
    # plt.colorbar()
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # for i in range(len(confusion_matrix)):
    #     for j in range(len(confusion_matrix[i])):
    #         plt.text(i, j, confusion_matrix[i][j])
    # plt.savefig('3.4.5.png')

    theta_hist, loss_hist = kernel_svm_subgrad_descent(X_train_vect, y_train, lambda_reg=1e-4, num_iter=200, batch_size=512)
    print(compute_accuracy(X_val_vect.dot(get_mapping_matrix(X_val_vect)), y_val, theta_hist[-1]))


if __name__ == '__main__':
    main()
