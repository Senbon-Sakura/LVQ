import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xlrd
import datetime

def loadDataSet(filename):
    workbook = xlrd.open_workbook(filename)
    worksheet = workbook.sheet_by_index(0)
    nrows = worksheet.nrows
    ncols = worksheet.ncols
    dataArr = []
    labelArr = []
    for i in range(nrows):
        rowVals = np.array(worksheet.row_values(i))
        dataArr.append(rowVals[1:-1])
        labelArr.append((rowVals[-1]))
    dataArr = np.array(dataArr)
    labelArr = np.array(labelArr)
    return dataArr, labelArr

def LVQ(dataArr, labelArr, K, eta=0.1, iters=10, epsilon=1e-6):
    sampNum, featNum = dataArr.shape
    # 初始化一组原型向量{p1, p2, ... , pK}, 必须保证每个class中抽取一个
    originClassNum = np.unique(labelArr)
    p = []
    for classVal in originClassNum:
        p.append(dataArr[np.where(labelArr == classVal)][0])
    pLabel = originClassNum
    for iter in range(iters):
        # 从样本集D随机选取样本（xj,yj)
        randNum = np.random.randint(0, sampNum)
        xj = dataArr[randNum]
        yj = labelArr[randNum]
        # 计算样本(xj, yj)到K个原型向量之间的距离
        diff = xj-p
        distance = sum(diff[:,i]**2 for i in range(featNum))
        # 找到最短距离
        minArg = np.argmin(distance)
        if yj == pLabel[minArg]:
            pNew = p[minArg] + eta * (xj - p[minArg])
        else:
            pNew = p[minArg] - eta * (xj - p[minArg])
        if (sum((pNew[i]-p[minArg][i])**2 for i in range(featNum))) > epsilon:
            p[minArg] = pNew
            print("LVQ algorithm in %d-th iteration" % (iter + 1))
        #else:
        #    print("LVQ algorithm converged in %d-th iteration" % (iter + 1))
        #    break
    return p, pLabel

def plotScatter(dataArr, p, pLabel, K):
    sampNum, featNum = dataArr.shape
    labelArr = np.zeros((sampNum))
    for i in range(sampNum):
        diff = dataArr[i] - p
        distance = sum(diff[:,j]**2 for j in range(featNum))
        labelArr[i] = np.argmin(distance)

    C = [ list() for i in range(K)]
    for i in range(K):
        C[i] = dataArr[np.where(labelArr == i)]

    plt.figure()
    for k in range(K):
        plt.scatter(C[k][:,0], C[k][:,1], c=list(mcolors.TABLEAU_COLORS)[k])
        plt.scatter(p[k][0], p[k][1], marker='*', c=list(mcolors.TABLEAU_COLORS)[k], s=400)
    curTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    figName = 'LVQ_' + curTime + '.png'
    plt.savefig(figName)
    plt.show()


#dataArr, labelArr = loadDataSet('watermelon 4.0.xlsx')
dataArr, labelArr = loadDataSet('testSet.xlsx')
K = 4
p, pLabel = LVQ(dataArr, labelArr, K, iters=50)
plotScatter(dataArr, p, pLabel, K)







