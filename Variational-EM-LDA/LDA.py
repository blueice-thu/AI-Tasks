import numpy as np
import codecs
import jieba
import re
import random
import math
from scipy.special import psi


# itemIdList : the list of distinct terms in the document
# itemCountList : the list of number of the existence of corresponding terms
# wordCount : the number of total words (not terms)
class Document:
    def __init__(self, itemIdList, itemCountList, wordCount):
        self.itemIdList = itemIdList
        self.itemCountList = itemCountList
        self.wordCount = wordCount


# preprocessing (segmentation, stopwords filtering, represent documents as objects of class Document)
def preprocessing():
    # read the list of stopwords
    file = codecs.open('stopwords.dic', 'r', 'utf-8')
    stopwords = [line.strip() for line in file]
    file.close()

    # read the corpus for training
    file = codecs.open('dataset.txt', 'r', 'utf-8')
    documents = [document.strip() for document in file]
    file.close()

    docs = []
    word2id = {}
    id2word = {}

    currentWordId = 0
    for document in documents:
        word2Count = {}
        # segmentation
        segList = jieba.cut(document)
        for word in segList:
            word = word.lower().strip()
            # filter the stopwords
            if len(word) > 1 and not re.search('[0-9]', word) and word not in stopwords:
                if word not in word2id:
                    word2id[word] = currentWordId
                    id2word[currentWordId] = word
                    currentWordId += 1
                if word in word2Count:
                    word2Count[word] += 1
                else:
                    word2Count[word] = 1
        itemIdList = []
        itemCountList = []
        wordCount = 0

        for word in word2Count.keys():
            itemIdList.append(word2id[word])
            itemCountList.append(word2Count[word])
            wordCount += word2Count[word]

        docs.append(Document(itemIdList, itemCountList, wordCount))

    return docs, word2id, id2word


def maxItemNum():
    num = 0
    for d in range(0, N):
        if len(docs[d].itemIdList) > num:
            num = len(docs[d].itemIdList)
    return num


def initialLdaModel():
    for z in range(0, K):
        for w in range(0, M):
            nzw[z, w] += 1.0 / M + random.random()
            nz[z] += nzw[z, w]
    updateVarphi()


# update model parameters : varphi (the update of alpha is ommited)
def updateVarphi():
    for z in range(0, K):
        for w in range(0, M):
            if nzw[z, w] > 0:
                varphi[z, w] = math.log(nzw[z, w]) - math.log(nz[z])
            else:
                varphi[z, w] = -100


# update variational parameters : gamma and phi
def variationalInference(docs: list, d: int, gamma: np.ndarray, phi: np.ndarray):
    ##########################################################################
    # update variational parameters : gamma and phi                          #
    #                                                                        #
    #                                                                        #
    #                                                                        #
    ##########################################################################
    # TODO
    # gamma: (16, 10)
    # phi: (104, 10)
    curr_doc = docs[d]
    n_word = len(curr_doc.itemIdList)
    psi_gamma = np.zeros(K)

    for k in range(K):
        gamma[d, k] = alpha + curr_doc.wordCount / K
        psi_gamma[k] = psi(gamma[d, k])
        for n in range(n_word):
            phi[n, k] = 1 / K

    for _ in range(iterInference):
        for n in range(n_word):
            phi_sum = 0
            phi_last = np.zeros(K)
            for k in range(K):
                phi_last[k] = phi[n, k]
                phi[n, k] = psi_gamma[k] + varphi[k, curr_doc.itemIdList[n]]
                phi_sum = phi[n, k] if k <= 0 else np.log(np.exp(phi_sum) + np.exp(phi[n, k]))
            for k in range(K):
                phi[n, k] = np.exp(phi[n, k] - phi_sum)
                gamma[d, k] = gamma[d, k] + curr_doc.itemCountList[n] * (phi[n, k] - phi_last[k])
                psi_gamma[k] = psi(gamma[d, k])


# calculate the gamma parameter of new document
def inferTopicOfNewDocument():
    testDocs = []
    # read the corpus to be inferred
    file = codecs.open('infer.txt', 'r', 'utf-8')
    testDocuments = [document.strip() for document in file]
    file.close()

    for d in range(0, len(testDocuments)):
        document = testDocuments[d]
        word2Count = {}
        # segmentation
        segList = jieba.cut(document)
        for word in segList:
            word = word.lower().strip()
            if word in word2id:
                if word in word2Count:
                    word2Count[word] += 1
                else:
                    word2Count[word] = 1

        itemIdList = []
        itemCountList = []
        wordCount = 0

        for word in word2Count.keys():
            itemIdList.append(word2id[word])
            itemCountList.append(word2Count[word])
            wordCount += word2Count[word]

        testDocs.append(Document(itemIdList, itemCountList, wordCount))

    gamma = np.zeros([len(testDocuments), K])
    for d in range(0, len(testDocs)):
        phi = np.zeros([len(testDocs[d].itemIdList), K])
        variationalInference(testDocs, d, gamma, phi)

    return gamma


docs, word2id, id2word = preprocessing()

# number of documents for training
N = len(docs)
# number of distinct terms
M = len(word2id)
# number of topic
K = 30
# iteration times of variational inference, judgment of the convergence by calculating likelihood is ommited
iterInference = 20
# iteration times of variational EM algorithm, judgment of the convergence by calculating likelihood is ommited
iterEM = 20

# initial value of hyperparameter alpha
alpha = 5
# sufficient statistic of alpha
alphaSS = 0
# the topic-word distribution (beta in D. Blei's paper)
varphi = np.zeros([K, M])
# topic-word count, this is a sufficient statistic to calculate varphi
nzw = np.zeros([K, M])
# topic count, sum of nzw with w ranging from [0, M-1], for calculating varphi
nz = np.zeros([K])

# inference parameter gamma
gamma = np.zeros([N, K])
# inference parameter phi
phi = np.zeros([maxItemNum(), K])

# initialization of the model parameter varphi, the update of alpha is ommited
initialLdaModel()

# variational EM Algorithm
for iteration in range(0, iterEM):
    nz = np.zeros([K])
    nzw = np.zeros([K, M])
    alphaSS = 0
    # E-Step
    for d in range(0, N):
        variationalInference(docs, d, gamma, phi)
        gammaSum = 0
        for z in range(0, K):
            gammaSum += gamma[d, z]
            alphaSS += psi(gamma[d, z])
        alphaSS -= K * psi(gammaSum)

        for w in range(0, len(docs[d].itemIdList)):
            for z in range(0, K):
                nzw[z][docs[d].itemIdList[w]] += docs[d].itemCountList[w] * phi[w, z]
                nz[z] += docs[d].itemCountList[w] * phi[w, z]

    # M-Step
    updateVarphi()

# calculate the top 10 terms of each topic
topicwords = []
maxTopicWordsNum = 10
for z in range(0, K):
    ids = varphi[z, :].argsort()
    topicword = []
    for j in ids:
        topicword.insert(0, id2word[j])
    topicwords.append(topicword[0: min(10, len(topicword))])

# infer the topic of each new document
inferGamma = inferTopicOfNewDocument()
inferZ = []
for i in range(0, len(inferGamma)):
    inferZ.append(inferGamma[i, :].argmax())

for topicword in topicwords:
    print(topicword)
print()
print(inferGamma)
