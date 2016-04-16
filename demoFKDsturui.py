'''
==============================
  Facial Keypoints Detection
  脸部关键点监测
==============================

Author：Stu. Rui
Version：v1.0 2016.4.3 20:41

'''
print(__doc__)

import pandas as pd
import numpy as np
import csv as csv
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_validation import cross_val_score, train_test_split

train_file = 'training.csv'         # 训练集数据
test_file = 'test.csv'              # 测试集数据 1783 张图片
test_type = 'IdLookupTable.csv'     # 测试集样表 行号, 图编号, 标签名

pd.set_option('chained_assignment',None)

#######################################################################################
# csv 数据读取，返回 df (pandas)
def csvFileRead(filename):

    print('Loading', filename)
    df = pd.read_csv(filename, header=0, encoding='GBK')
    print('Loaded')

    # 缺失项数据删除
    if 'train' in filename:
        df = df.dropna()
    ''' 数据查看
    print('\n数据表尺寸: ', df.values.shape)
    print('类别统计：\n')
    print(df.count(), '\n') 
    '''
    return df

# 结果存储
def csvSave(filename, ids, predicted):
    with open(filename, 'w') as mycsv:
        mywriter = csv.writer(mycsv)
        mywriter.writerow(['RowId','Location'])
        mywriter.writerows(zip(ids, predicted))

# 训练集数据预处理
def preTrain():
    
    print('-----------------Training reading...-----------------')
    df = csvFileRead(train_file)
    
    print('Image: str -> narray')
    df.Image = df.Image.apply(lambda im: np.fromstring(im, sep=' '))
    print('Image transfered.\n')

    # problem: 7049*9046 MemoryError -> df.dropna()
    X = np.vstack(df.Image.values) / 255.
    X.astype(np.float32)

    y = df[df.columns[:-1]].values
    y = (y-48)/48.
    y = y.astype(np.float32)
    '''
    # 加入人工镜像图片
    print('加入人工镜像图片...')
    X, y = imageSym(X, y)
    '''
    X, y = shuffle(X, y, random_state=42)

    yd = dict()
    for i in range(len(df.columns[:-1].values)):
        yd[df.columns[i]] = i

    return X, y, yd

# 预测集数据预处理
def preTest():
    print('-----------------Test reading...-----------------')
    df = csvFileRead(test_file)

    print('Image: str -> narray')
    df.Image = df.Image.apply(lambda im: np.fromstring(im, sep=' '))
    print('Image transfered.\n')
    # 测试集图像
    X = np.vstack(df.Image.values) / 255.
    X.astype(np.float32)

    # 预测内容：行号, 图编号, 标签名
    df = csvFileRead(test_type)
    RowId = df.RowId.values
    ImageId = df.ImageId.values - 1
    FeatureName = df.FeatureName.values

    return RowId, ImageId, FeatureName, X

# 人工特征：镜像图片
def imageSym(X, y):
    nX = np.zeros(X.shape)
    ny = np.zeros(y.shape)
    for i in range(X.shape[0]):
        temp = X[i,:].reshape(96, 96)
        temp = temp[:,::-1]
        nX[i,:] = temp.reshape(-1)
        ny[i,0::2] = -y[i,0::2]
        ny[i,1::2] = y[i,1::2]
    X = np.vstack((X, nX))
    y = np.vstack((y, ny))
    return X, y
        

#######################################################################################
# 数据拟合，
# There are so many about regressioners hyperparameters and complexity.
# We can see 30 position to 1 regressioners that cost long long time.
# Thus, these problems seemed too hard to work out,
# maybe I should try GPU or AWS. So confused but amazd.


# 30 个拟合器进行拟合
def modelfit(train_X, train_y, test_X, yd, ImageId, FeatureName):
    ################################### There are fitting codes.
    

    # 30 个拟合器对应 1 个位置
    n_clf = 30
    clfs = [
        KernelRidge(kernel='rbf', gamma=2e-4, alpha=1e-2), KernelRidge(kernel='rbf', gamma=2e-4, alpha=1e-2),
        KernelRidge(kernel='rbf', gamma=2e-4, alpha=1e-2), KernelRidge(kernel='rbf', gamma=2e-4, alpha=1e-2),
        KernelRidge(kernel='rbf', gamma=2e-4, alpha=1e-2), KernelRidge(kernel='rbf', gamma=2e-4, alpha=1e-2),
        KernelRidge(kernel='rbf', gamma=2e-4, alpha=1e-2), KernelRidge(kernel='rbf', gamma=2e-4, alpha=1e-2),
        KernelRidge(kernel='rbf', gamma=2e-4, alpha=1e-2), KernelRidge(kernel='rbf', gamma=2e-4, alpha=1e-2),
        KernelRidge(kernel='rbf', gamma=2e-4, alpha=1e-2), KernelRidge(kernel='rbf', gamma=2e-4, alpha=1e-2),
        KernelRidge(kernel='rbf', gamma=2e-4, alpha=1e-2), KernelRidge(kernel='rbf', gamma=2e-4, alpha=1e-2),
        KernelRidge(kernel='rbf', gamma=2e-4, alpha=1e-2), KernelRidge(kernel='rbf', gamma=2e-4, alpha=1e-2),
        KernelRidge(kernel='rbf', gamma=2e-4, alpha=1e-2), KernelRidge(kernel='rbf', gamma=2e-4, alpha=1e-2),
        KernelRidge(kernel='rbf', gamma=2e-4, alpha=1e-2), KernelRidge(kernel='rbf', gamma=2e-4, alpha=1e-2),
        KernelRidge(kernel='rbf', gamma=2e-4, alpha=1e-2), KernelRidge(kernel='rbf', gamma=2e-4, alpha=1e-2),
        KernelRidge(kernel='rbf', gamma=2e-4, alpha=1e-2), KernelRidge(kernel='rbf', gamma=2e-4, alpha=1e-2),
        KernelRidge(kernel='rbf', gamma=2e-4, alpha=1e-2), KernelRidge(kernel='rbf', gamma=2e-4, alpha=1e-2),
        KernelRidge(kernel='rbf', gamma=2e-4, alpha=1e-2), KernelRidge(kernel='rbf', gamma=2e-4, alpha=1e-2),
        KernelRidge(kernel='rbf', gamma=2e-4, alpha=1e-2), KernelRidge(kernel='rbf', gamma=2e-4, alpha=1e-2)]
    
    print('-----------------开始训练...------------------')
    # 超参数
    C_para = np.logspace(-2, 4, 7)      # SVR.C
    G_para = np.logspace(-4, -3, 6)     # kernel = 'rbf'.gamma
    A_para = np.logspace(-3, 1, 5)      # KernelRidge.alpha
    # 训练
    for i in range(n_clf):
        print('Training', i, 'clf...')
        clfs[i].fit(train_X, train_y[:,i])
    # 打印训练误差
    predict = np.zeros([train_y.shape[0], 30]).astype(np.float32)
    for i in range(n_clf):
        predict[:,i] = clfs[i].predict(train_X)
    print(calbais(predict, train_y))
    print()
    
    print('-----------------开始预测...------------------')
    # 预测
    pred = np.zeros([test_X.shape[0], 30]).astype(np.float32)
    for i in range(n_clf):
        pred[:,i] = clfs[i].predict(test_X)
    predicted = np.zeros(len(FeatureName))
    for i in range(len(FeatureName)):
        if i % 500 == 0:
            print('i =', i)
        else:
            pass
        imageID = ImageId[i]
        clfID = yd[FeatureName[i]]
        predicted[i] = pred[imageID, clfID]
    predicted = predicted*48.+48.
    
    return predicted

# 单一拟合器，同时对 30 个标签做拟合
def modelfitOne(train_X, train_y, test_X, yd, ImageId, FeatureName):
    n_clf = 1
    # 拟合器
    clf = KernelRidge(kernel='rbf', gamma=6e-4, alpha=2e-2)
    # 训练
    print('-----------------开始训练...------------------')
    clf.fit(train_X, train_y)
    # 预测
    print('-----------------开始预测...------------------')
    pred = clf.predict(test_X)
    predicted = np.zeros(len(FeatureName))
    for i in range(len(FeatureName)):
        if i % 500 == 0:
            print('i =', i)
        else:
            pass
        imageID = ImageId[i]
        clfID = yd[FeatureName[i]]
        predicted[i] = pred[imageID, clfID]
    predicted = predicted*48.+48.
    return predicted
    
# 均方根计算方法
def calbais(pred, y2):
    y_diff = pred - y2
    y_diff = y_diff.reshape(-1)
    sco = np.linalg.norm(y_diff)/(len(y2)**0.5)
    return sco
#######################################################################################
# 参数选择的调试函数
# 超参数调试 X-y
def testfit(clf, train_X, train_y):
    scores = list()
    for i in range(3):
        X1, X2, y1, y2 = train_test_split(train_X, train_y, test_size=0.3, random_state=42)
        pred = clf.fit(X1, y1).predict(X2)
        sco = calbais(pred, y2)
        scores.append(sco)
    print(scores)
    
# 测试图
def plotface(x, y):
    img = x.reshape(96, 96)
    plt.imshow(img, cmap='gray')
    y = y * 48 + 48
    plt.scatter(y[0::2], y[1::2], marker='x', s=20)
    plt.show()
        

#df = csvFileRead(train_file)
# 训练集数据读取
train_X, train_y, yd = preTrain()
# 测试集数据读取
RowId, ImageId, FeatureName, test_X = preTest()
# 1) 数据拟合: 30 个拟合器
#predicted = modelfit(train_X, train_y, test_X, yd, ImageId, FeatureName)
# 2) 数据拟合: 1 个拟合器
predicted = modelfitOne(train_X, train_y, test_X, yd, ImageId, FeatureName)
# 结果存储
csvSave('KernelRidge.csv', np.linspace(1, len(predicted), len(predicted)).astype(int), predicted)
