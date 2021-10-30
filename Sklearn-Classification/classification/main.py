import time, warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from MyKNNClassifier import MyKNNClassifier

warnings.filterwarnings("ignore")

columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
           "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
           "hours-per-week", "region", "salary"]

continuous_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
nominal_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'region']
label_column = 'salary'


class Encoder:
    """
    封装编码相关函数以统一编码方式，为每个属性分配一个 LabelEncoder
    """
    def __init__(self):
        """
        预先为每个离散属性值分配一个 Label 值
        """
        self.trans = {}
        for col in nominal_columns:
            self.trans[col] = LabelEncoder()
        self.trans['workclass'].fit(
            ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay",
             "Never-worked"])
        self.trans['education'].fit(
            ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th",
             "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"])
        self.trans['marital-status'].fit(
            ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent",
             "Married-AF-spouse"])
        self.trans['occupation'].fit(
            ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty",
             "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving",
             "Priv-house-serv", "Protective-serv", "Armed-Forces"])
        self.trans['relationship'].fit(["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"])
        self.trans['race'].fit(["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"])
        self.trans['sex'].fit(["Female", "Male"])
        self.trans['region'].fit(
            ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)",
             "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland",
             "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador",
             "China-Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand",
             "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"])

    def encode_x(self, X: pd.DataFrame):
        # 用 LabelEncoder 进行编码
        for col in nominal_columns:
            X[col] = self.trans[col].transform(X[col])

        # Minmax 归一化
        X['age'] /= 100
        X['workclass'] /= 8
        X['fnlwgt'] = np.log10(X['fnlwgt'] + 1) / np.log10(2e6)  # 对数缩放后归一化
        X['education'] /= 16
        X['education-num'] /= 16
        X['marital-status'] /= 14
        X['relationship'] /= 6
        X['race'] /= 5
        X['capital-gain'] = np.log10(X['capital-gain'] + 1) / 5
        X['capital-loss'] = np.log10(X['capital-loss'] + 1) / np.log10(5000)
        X['hours-per-week'] /= 100
        X['region'] /= 41

        return X

    @staticmethod
    def encode_y(y: pd.DataFrame):
        """
        '<=50k' --> 0, '>50K' --> 1
        """
        return y.replace('<=50K', 0).replace('>50K', 1).replace('<=50K.', 0).replace('>50K.', 1)

    def encode(self, X, Y):
        return self.encode_x(X), self.encode_y(Y)


def get_data(filename):
    """
    读取数据文件，处理缺失值，划分 X 和 Y

    :param filename: train.txt 和 test.txt 路径
    :return: [ X: pd.DataFrame, Y: pd.DataFrame ]
    """
    data = pd.read_table(filename, sep=', ', header=None, names=columns, engine='python')

    # 丢弃含有缺失数据的行
    data = data[data['workclass'] != '?']
    data = data[data['occupation'] != '?']
    data = data[data['region'] != '?']

    # 以下注释代码用于概览数据分布，生成分布图像，结果见 result/distribute 文件夹
    # print(data.describe())
    # for col in continuous_columns:
    #     x = [col] * data.shape[0]
    #     y = data[col].values.tolist()
    #     plt.plot(x, y, 'bo', ms=0.1)
    #     plt.savefig('./result/distribute/{}.png'.format(col))
    #     plt.close()
    # print(data[data.isnull().T.any()])

    return data.drop([label_column], axis=1), data[label_column]


def select_knn_n_neighbors(x_train, y_train):
    """
    使用 10 折交叉验证为 KNeighborsClassifier 模型挑选参数 n_neighbors

    :return: 得分最高的参数 n_neighbors
    """
    best_score = 0
    best_n = 0
    x, y = [], []
    for n_neighbors in range(1, 50):
        print('\rtesting n_neighbors = {}...'.format(n_neighbors), end='')

        model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=4)
        # 使用 sklearn 自带的评估函数进行 10 折交叉验证
        score = cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy').mean()
        x.append(n_neighbors)
        y.append(score)
        if score > best_score:
            best_score = score
            best_n = n_neighbors
    print('\rBest n_neighbors = {}'.format(best_n))

    # 绘制得分 - 参数值图像
    plt.xlabel('n_neighbors')
    plt.ylabel('mean cross validation score')
    plt.title('KNeighborsClassifier')
    plt.plot(x, y, color='red')
    plt.savefig('./result/KNN.png')  # 保存路径

    return best_n


def select_C(model, scope, x_train, y_train, model_name, max_iter=8000):
    """
    使用 10 折交叉验证为 LogisticRegression 和 LinearSVC 模型挑选惩罚参数 C

    :param model: 模型，只能为 LogisticRegression 或 LinearSVC
    :param scope: 挑选的参数范围
    :param x_train: 自变量数据
    :param y_train: 标签数据
    :param model_name: 模型名称，用于输出图像
    :param max_iter: 最大迭代次数
    :return: 得分最高的参数 C
    """
    best_score = 0
    best_C = 0
    x, y = [], []
    for C in scope:
        print('\rtesting C = {}...'.format(C), end='')

        selected_model = model(C=C, max_iter=max_iter)
        # 使用 sklearn 自带的评估函数进行 10 折交叉验证
        score = cross_val_score(selected_model, x_train, y_train, cv=10, scoring='accuracy').mean()
        x.append(C)
        y.append(score)
        if score > best_score:
            best_score = score
            best_C = C
    print('\rBest C = {}'.format(best_C))

    # 绘制得分 - 参数值图像
    plt.xlabel('C')
    plt.ylabel('mean cross validation score')
    plt.title(model_name)
    plt.plot(x, y, color='red')
    plt.savefig('./result/{}.png'.format(model_name))

    return best_C


def get_model(model_name: str, x_train, y_train):
    """
    选择使用的模型，并自动进行交叉验证、选取超参数

    :param model_name: 可供选择：Logistic, KNN, LinearSVC, MyKNN
    :param x_train: 自变量数据
    :param y_train: 标签数据
    :return: 选择的模型，已经设定好超参数
    """
    if model_name == 'Logistic':
        C = 2.04  # 代码中已经写好了最佳的超参数
        if C < 0:  # 将 C 改为负数值即可重新选取超参数（其他模型同理）
            scope = list(range(1, 201))
            scope = [i / 50 for i in scope]  # 选取范围：[0.02, 4]间均匀分布的 200 个点
            C = select_C(LogisticRegression, scope, x_train, y_train, model_name)
        return LogisticRegression(C=C)
    elif model_name == 'KNN':
        n_neighbors = 16
        if n_neighbors < 1:
            n_neighbors = select_knn_n_neighbors(x_train, y_train)
        return KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=4)
    elif model_name == 'LinearSVC':
        C = 0.12
        if C < 0:
            scope = list(range(1, 51))
            scope = [i / 50 for i in scope]
            C = select_C(LinearSVC, scope, x_train, y_train, model_name)
        return LinearSVC(C=C, max_iter=1e4)
    elif model_name == 'MyKNN':
        # 自己实现的 KNN 算法，与 KNeighborsClassifier 原理相同，不再重新选取参数
        # 但是算法效率很低，所以可以用 KNeighborsClassifier 选取参数之后直接来用
        return MyKNNClassifier(16)
    else:
        raise NotImplementedError('No such model: {}'.format(model_name))


def train(model, x_train, y_train):
    start = time.time()
    model.fit(x_train, y_train)
    end = time.time()

    print('Train time: %.2fs' % (end - start))


def test(model, x_test):
    start = time.time()
    y_pred = model.predict(x_test)
    end = time.time()

    print('Test time: %.2fs' % (end - start))

    return y_pred


def evaluate(y_true, y_pred):
    print("================= Test Result =================")
    print("Accuracy:        %.4f" % (metrics.accuracy_score(y_true, y_pred)))
    print("Recall:          %.4f" % (metrics.recall_score(y_true, y_pred)))
    print("Precision:       %.4f" % (metrics.precision_score(y_true, y_pred)))
    print("F1-score:        %.4f" % (metrics.f1_score(y_true, y_pred)))
    print("Cross Entropy:   %.4f" % metrics.log_loss(y_true, y_pred))
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_true, y_pred))

    show_confusion_matrix(y_true, y_pred)


def show_confusion_matrix(y_test, y_pred, model_name=''):
    """
    绘制混淆矩阵

    :param model_name: 不为空时，可将混淆矩阵存储到 result/confusion_matrix 文件夹
    """
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

    plt.imshow(confusion_matrix, cmap=plt.cm.Blues)
    indices = range(len(confusion_matrix))
    plt.xticks(indices, [0, 1])
    plt.yticks(indices, [0, 1])
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix[i])):
            plt.text(i, j, confusion_matrix[i][j])

    if model_name:
        plt.title(model_name)
        plt.savefig('./result/confusion_matrix/{}.png'.format(model_name))
    plt.show()


if __name__ == '__main__':
    # 可供选择的模型名称：{ Logistic, KNN, LinearSVC, MyKNN }
    model_name = 'Logistic'
    encoder = Encoder()

    x_train, y_train = get_data('./data/train.txt')
    x_test, y_test = get_data('./data/test.txt')

    x_train, y_train = encoder.encode(x_train, y_train)
    x_test, y_test = encoder.encode(x_test, y_test)

    print('train_data: {}'.format(x_train.shape))
    print('test_data: {}'.format(x_test.shape))

    model = get_model(model_name, x_train, y_train)
    train(model, x_train, y_train)
    y_pred = test(model, x_test)
    evaluate(y_test, y_pred)
