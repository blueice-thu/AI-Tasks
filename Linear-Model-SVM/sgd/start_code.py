from numpy.core.fromnumeric import size
import pandas as pd
import logging
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import axis0_safe_slice


def feature_normalization(train: np.array, test: np.array):
    """将训练集中的所有特征值映射至[0,1]，对验证集上的每个特征也需要使用相同的仿射变换

    Args：
        train - 训练集，一个大小为 (num_instances, num_features) 的二维 numpy 数组
        test - 测试集，一个大小为 (num_instances, num_features) 的二维 numpy 数组
    Return：
        train_normalized - 归一化后的训练集
        test_normalized - 标准化后的测试集

    """
    # TODO 2.2.1
    train_min = train.min(axis=0)
    train_max = train.max(axis=0)
    train_normalized = (train - train_min) / (train_max - train_min)
    test_normalized = (test - train_min) / (train_max - train_min)
    return train_normalized, test_normalized


def compute_square_loss(X, y, theta):
    """
    给定一组 X, y, theta，计算用 X*theta 预测 y 的平方损失

    Args：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        theta - 参数向量，数组大小 (num_features)

    Return：
        loss - 平方损失，标量
    """
    # TODO 2.2.5
    loss = y - np.dot(X, theta)
    loss = np.dot(loss.T, loss) / (X.shape[0])
    return loss


def compute_square_loss_gradient(X, y, theta):
    """
    计算theta处平方损失的梯度。

    Args：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        theta - 参数向量，数组大小（num_features）

    Return：
        grad - 梯度向量，数组大小（num_features）
    """
    # TODO 2.2.6
    grad = 2 * np.dot(X.T, np.dot(X, theta) - y) / X.shape[0]
    return grad


def grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4):
    """梯度检查
    如果实际梯度和近似梯度的欧几里得距离超过容差，则梯度计算不正确。

    Args：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        theta - 参数向量，数组大小（num_features）
        epsilon - 容差

    Return：
        梯度是否正确

    """
    true_gradient = compute_square_loss_gradient(X, y, theta)  # the true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features)  # Initialize the gradient we approximate
    # TODO 2.2.7
    for i in range(num_features):
        h = np.zeros(num_features)
        h[i] = 1
        approx_grad[i] = (compute_square_loss(X, y, theta + epsilon * h) - compute_square_loss(X, y, theta - epsilon * h)) / (2 * epsilon)
    distance = np.linalg.norm(true_gradient - approx_grad)
    return distance <= tolerance


def batch_grad_descent(X, y, alpha=0.1, num_iter=1000, check_gradient=False):
    """
    全批量梯度下降算法

    Args：
        X - 特征向量， 数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        alpha - 梯度下降的步长
        num_iter - 要运行的迭代次数
        check_gradient - 更新时是否检查梯度

    Return：
        theta_hist - 存储迭代中参数向量的历史，大小为 (num_iter+1, num_features) 的二维 numpy 数组
        loss_hist - 目标函数向量的历史，大小为 (num_iter+1) 的一维 numpy 数组
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #initialize loss_hist
    theta = np.zeros(num_features) #initialize theta
    # TODO 2.2.8
    theta_hist[0] = theta
    loss_hist[0] = compute_square_loss(X, y, theta)
    for iter in range(1, num_iter + 1):
        loss_gradient = compute_square_loss_gradient(X, y, theta)

        if check_gradient and not grad_checker(X, y, theta):
            print("grad_checker failed!")
            break
        
        theta = theta - alpha * loss_gradient

        loss_value = compute_square_loss(X, y, theta)
        theta_hist[iter] = theta
        loss_hist[iter] = loss_value

    return theta_hist, loss_hist


def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    """
    计算岭回归损失函数的梯度

    参数：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        theta - 参数向量，数组大小（num_features）
        lambda_reg - 正则化系数

    返回：
        grad - 梯度向量，数组大小（num_features）
    """
    # TODO 2.3.2
    grad = 2 * np.dot(X.T, np.dot(X, theta) - y) / X.shape[0] + 2 * lambda_reg * theta
    return grad


def regularized_grad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000):
    """
    参数：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        alpha - 梯度下降的步长
        lambda_reg - 正则化系数
        numIter - 要运行的迭代次数

    返回：
        theta_hist - 参数向量的历史，数组大小 (num_iter+1, num_features)
        loss_hist - 没有正则化项的损失函数的历史，一维 numpy 数组。
    """
    (num_instances, num_features) = X.shape
    theta = np.zeros(num_features)  # Initialize theta
    theta_hist = np.zeros((num_iter+1, num_features))  # Initialize theta_hist
    loss_hist = np.zeros(num_iter+1)  # Initialize loss_hist
    # TODO 2.3.3
    theta_hist[0] = theta
    loss_hist[0] = compute_square_loss(X, y, theta)
    for iter in range(1, num_iter + 1):
        loss_gradient = compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)
        theta = theta - alpha * loss_gradient

        loss_value = compute_square_loss(X, y, theta)
        theta_hist[iter] = theta
        loss_hist[iter] = loss_value
        if iter % 100 == 0:
            print("iter {}: loss = {}".format(iter, loss_value))

    return theta_hist, loss_hist


def stochastic_grad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000, batch_size=1):
    """
    随机梯度下降

    参数：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        alpha - 字符串或浮点数。梯度下降步长
                注意：在 SGD 中，使用固定步长并不总是一个好主意。通常设置为 1/sqrt(t) 或 1/t
                如果 alpha 是一个浮点数，那么每次迭代的步长都是 alpha。
                如果 alpha == "1/sqrt(t)", alpha = 1/sqrt(t)
                如果 alpha == "1/t", alpha = 1/t
        lambda_reg - 正则化系数
        num_iter - 要运行的迭代次数
        batch_size - 批大小

    返回：
        theta_hist - 参数向量的历史，大小的 2D numpy 数组 (num_iter+1, num_features)
        loss hist - 正则化损失函数向量的历史，数组大小(num_iter+1)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features)  # Initialize theta
    theta_hist = np.zeros((num_iter+1, num_features))  # Initialize theta_hist
    loss_hist = np.zeros(num_iter+1)  # Initialize loss_hist
    # TODO 2.4.3, 2.4.4
    theta_hist[0] = theta
    loss_hist[0] = compute_square_loss(X, y, theta)
    for iter in range(1, num_iter + 1):
        sample_index = np.random.randint(num_instances, size=batch_size)
        X_sample, y_sample = X[sample_index], y[sample_index]
        loss_gradient = compute_regularized_square_loss_gradient(X_sample, y_sample, theta, lambda_reg)

        if isinstance(alpha, float):
            theta = theta - alpha * loss_gradient
        elif alpha == "1/sqrt(t)":
            theta = theta - 1 / np.sqrt(num_iter) * loss_gradient
        elif alpha == "1/t":
            theta = theta - 1 / num_iter * loss_gradient
        else:
            print("Unknown alpha define: " + str(alpha))
            exit(-1)

        loss_value = compute_square_loss(X, y, theta)
        theta_hist[iter] = theta
        loss_hist[iter] = loss_value
        if iter % 100 == 0:
            print("iter {}: loss = {}".format(iter, loss_value))

    return theta_hist, loss_hist


def main():
    # 加载数据集
    print('loading the dataset')

    df = pd.read_csv(r'D:\SomeCodes\ml-assignment1\sgd\data.csv', delimiter=',')
    # df = pd.read_csv('data.csv', delimiter=',')
    X = df.values[:, :-1]
    y = df.values[:, -1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # 增加偏置项
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))  # 增加偏置项

    # TODO
    # 2.2.9
    num_iter = 1500
    # for alpha in [0.05, 0.03, 0.01]:
    #     _, loss_hist = batch_grad_descent(X_train, y_train, alpha=alpha, num_iter=num_iter, check_gradient=False)
    #     plt.plot(np.arange(num_iter + 1), loss_hist, linestyle='-', label=str(alpha))
    # plt.legend()
    # plt.xlabel('iter')
    # plt.ylabel('loss')
    # plt.savefig("2.2.9.png")

    # 2.3.4
    # theta_list, loss_list = regularized_grad_descent(X_train, y_train, alpha=0.01, lambda_reg=100, num_iter=num_iter)
    # test_loss = compute_square_loss(X_test, y_test, theta_list[-1])
    # print(test_loss)

    # 2.4.3
    # for batch_size in [4, 16, 64, 128, 256]:
    #     _, loss_hist = stochastic_grad_descent(X_train, y_train, alpha="1/sqrt(t)", lambda_reg=1e-3, num_iter=num_iter, batch_size=batch_size)
    #     plt.plot(np.arange(num_iter + 1)[20:][::15], loss_hist[20:][::15], linestyle='-', label=str(batch_size))
    # plt.legend()
    # plt.xlabel('iter')
    # plt.ylabel('loss')
    # plt.savefig("2.4.4.png")


if __name__ == "__main__":
    main()
