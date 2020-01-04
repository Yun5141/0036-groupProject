# ------------------- import packages -----------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from os import chdir

# !pip install sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve

import io

# !pip install geopy
from geopy.distance import geodesic 
from geopy.distance import great_circle 

import time
import datetime as dt
from datetime import datetime

import math

from collections import deque

# ------------------- import data -------------------------
url="https://raw.githubusercontent.com/Yun5141/comp0036/master/elp_up-to-date_data.csv"
training_data=pd.read_csv(url)
url = "https://raw.githubusercontent.com/Yun5141/comp0036/master/stadiums-with-GPS-coordinates.csv"
geometricData = pd.read_csv(url)


# ------------------ helper functions -------------------
# ********************************************
# to remove data that contains None, NaN, infinite or overflowed
def removeInvalidData(data):

    # remove data which contains None
    data.dropna(axis=0, how='any',inplace=True)

    # remove data which contains NaN, infinite or overflowed number 
    indices_to_keep = ~data.isin([np.nan, np.inf, -np.inf]).any(1)
    data = data[indices_to_keep]

    return data

# !!! 【report中: 第一步先检查有无空值】
assert training_data.shape[0] == removeInvalidData(training_data).shape[0]
#result: there is no empty value at the initial stage 

# ********************************************
# unify the different date formats and convert the type from str to timestamp   [done]
def unifyDate(data):

    if not isinstance(data.Date[0],str):
        return

    newDate = []
    for _, matchInfo in data.iterrows():
        if len(matchInfo.Date) == 8 :
            newDate.append( pd.to_datetime(matchInfo.Date, format="%d/%m/%y" ))
        elif len(matchInfo.Date) == 9 :
            newDate.append( pd.to_datetime(matchInfo.Date, format="%d %b %y" ))
        elif len(matchInfo.Date) == 10 :
            newDate.append(  pd.to_datetime(matchInfo.Date, format="%d/%m/%Y" ))
    
    data['Date'] = pd.Series(newDate).values

#unifyDate(training_data)

# ------------------ Inital Data Exploration -------------------  
# ********************************************
# to see the number of matches each year / season
def separateData(data):
    dataframe_collection = {}

    for year in range(2008, 2020):
        dataframe_collection[year] = data[(data.Date > dt.datetime(year,8,1,0,0) ) & (data.Date < dt.datetime(year+1, 6, 1,0,0))]

    return dataframe_collection

'''
data = separateData(training_data)
for key in data.keys():
    print("\n" +"="*40)
    print(key)
    print("-"*40)
    print(data[key])
'''
#result: 380 rows * 11 dataframes + 170 rows * 1 dataframes = 4350 rows

# ********************************************
# !!! 【在数据正式处理前后各用一次这个函数，即两次的data exploration section】
def checkAverageWinRate(data, resultWinner):

    if resultWinner not in ['H', 'A', 'D']:
        raise Exception('The second argument should only take values within [“H”,“A”,“D”]')
    
    n_wins = len(data[data.FTR == resultWinner])

    return n_wins / data.shape[0]

#prediction = checkAverageWinRate(training_data, 'H')

#results of raw data (ie, when nothing applied to the training data): 
#total number of matches = 4350
#home team = 0.4606896551724138 ~ 0.461; 
#away team = 0.2910344827586207 ~ 0.291;
#draw = 0.2482758620689655 ～ 0.248

# ------------------- Feature Construction ------------------——————————
#*******************************
# get the distance needed to travel for the away team   [done] 
def getDistance(data, geometricData):
  array = []
  for x in data.iterrows():
   
    home_lat = (geometricData.loc[geometricData['Team'] == x[1].HomeTeam]).Latitude
    home_long = (geometricData.loc[geometricData['Team'] == x[1].HomeTeam]).Longitude
    home_location = (np.float32(home_lat), np.float32(home_long))
    
    away_lat = (geometricData.loc[geometricData['Team'] == x[1].AwayTeam]).Latitude
   
    away_long = (geometricData.loc[geometricData['Team'] == x[1].AwayTeam]).Longitude
    away_location = (np.float32(away_lat), np.float32(away_long))
    array.append(np.float32(geodesic(home_location, away_location).km))
  
  
  DIS = pd.Series(array)
  data.loc[:,'DIS'] = DIS

  return data

#getDistance(training_data, geometricData) 
#print(training_data)

#*******************************
# get match week    [done]
def getMW(data, startYear):  
    MW = []
    Flag = 0
    year = startYear

    for _, matchInfo in data.iterrows():
        checkYear = (matchInfo.Date > dt.datetime(year,8,1,0,0)) & (matchInfo.Date < dt.datetime(year+1, 6, 1,0,0)) 
        
        if not checkYear:
            year += 1
            Flag = 0
    
        if (Flag == 0):
            firstDate = matchInfo.Date
            Flag = 1

        week = (matchInfo.Date - firstDate).days // 7 +1
        
        MW.append(week) 

    data.loc[:,'MW'] = pd.Series(MW).values

    return data

'''
使用方法1: 赛季分开成12张分表，则
    unifyDateformat(data)
    separate(data)
然后
for key in data.keys():
    print("\n" +"="*40)
    print(key)
    print("-"*40)
    #print(data[key])
    print(getMW(data[key], key))

使用方法2: 不分赛季，使用完整的表，则
    unifyDateFormat(data)
然后
print(getMW(data, 2008))
'''

#*******************************
# calculate the delta time from last match for home team and away team  [done]
def getDeltaTime(data):
    
    teams = {}

    HDT = []
    ADT = []

    for i in range(len(data)):
        if (i % 380 == 0):
            for name in data.groupby('HomeTeam').mean().T.columns:
                teams[name] = []    # to store last match date

        currentDate = data.iloc[i].Date

        try:
            homeLastMatchDate = teams[data.iloc[i].HomeTeam].pop()
            awayLastMatchDate = teams[data.iloc[i].AwayTeam].pop()
        except:
            homeLastMatchDate = currentDate
            awayLastMatchDate = currentDate

        hdt = currentDate - homeLastMatchDate
        adt = currentDate - awayLastMatchDate

        HDT.append(hdt.days)
        ADT.append(adt.days)

        teams[data.iloc[i].HomeTeam].append(currentDate)
        teams[data.iloc[i].AwayTeam].append(currentDate)

    data.loc[:,'HDT'] = HDT
    data.loc[:,'ADT'] = ADT

    return data

#unifyDateFormat(training_data)
#getMW(training_data,2008)
#getDeltaTime(training_data)
#training_data.loc[377:400,["Date","HomeTeam","AwayTeam","MW","HDT","ADT"]]

#*****************************
# calculate the cumulative goal difference (before this match) scored by home team and away team    [done]
def getCumulativeGoalsDiff(data):
    teams = {}
    HCGD = [] 
    ACGD = []   

    # for each match
    for i in range(len(data)):
        
        if (i % 380 == 0):
            for name in data.groupby('HomeTeam').mean().T.columns:
                teams[name] = []

        FTHG = data.iloc[i]['FTHG']
        FTAG = data.iloc[i]['FTAG']

        try:
            cgd_h = teams[data.iloc[i].HomeTeam].pop()
            cgd_a = teams[data.iloc[i].AwayTeam].pop()
        except:
            cgd_h = 0
            cgd_a = 0

        HCGD.append(cgd_h)
        ACGD.append(cgd_a)
        cgd_h = cgd_h + FTHG - FTAG
        teams[data.iloc[i].HomeTeam].append(cgd_h)
        cgd_a = cgd_a + FTAG - FTHG
        teams[data.iloc[i].AwayTeam].append(cgd_a)

    data.loc[:,'HCGD'] = HCGD
    data.loc[:,'ACGD'] = ACGD
    return data

#getCumulativeGoalsDiff(training_data)
#training_data

#****************************
# get average goal difference per week
def getAverageGD(data):

    data.eval('HAGD = HCGD / MW', inplace=True)
    data.eval('AAGD = ACGD / MW', inplace=True)

    return data

# !!!【必须有了CGD与MW之后再写这一个；在第二次explore画图时舍弃CGD，AGD其中一个】
# unifyDateFormat(training_data)
# getMW(training_data,2008)
# getCumulativeGoalsDiff(training_data)
# getAverageGD(training_data)

#****************************
# !!!【 写report时在代码块外提一句: 因为在最开始用separateData()已发现，每年比赛数都是固定的380场，所以循环里可直接用i%380==0来初始化】
# 统计每支队伍最近三场比赛的表现    [done]
def getPerformanceOfLast3Matches(data):
    HM1 = []    # result of the last match of home team
    AM1 = []    # result of the last match of away team

    HM2 = []    # result of the 2nd last match of home team
    AM2 = []

    HM3 = []    # result of the 3rd last match of home team
    AM3 = []

    teams = {}

    for i in range(len(data)):
        
        if (i % 380 == 0):
            for name in data.groupby('HomeTeam').mean().T.columns:
                teams[name] = deque([None, None, None])  #[3rd, 2nd, latest data]

        HM3.append(teams[data.iloc[i].HomeTeam].popleft())
        AM3.append(teams[data.iloc[i].AwayTeam].popleft())
        HM2.append(teams[data.iloc[i].HomeTeam][0])
        AM2.append(teams[data.iloc[i].AwayTeam][0])
        HM1.append(teams[data.iloc[i].HomeTeam][1])
        AM1.append(teams[data.iloc[i].AwayTeam][1])

        if data.iloc[i].FTR == 'H':
            # 主场 赢，则主场记为赢，客场记为输
            teams[data.iloc[i].HomeTeam].append('W')
            teams[data.iloc[i].AwayTeam].append('L')
        elif data.iloc[i].FTR == 'A':
            # 客场 赢，则主场记为输，客场记为赢
            teams[data.iloc[i].AwayTeam].append('W')
            teams[data.iloc[i].HomeTeam].append('L')
        else:
            # 平局
            teams[data.iloc[i].AwayTeam].append('D')
            teams[data.iloc[i].HomeTeam].append('D')

    data.loc[:,'HM1'] = HM1
    data.loc[:,'AM1'] = AM1
    data.loc[:,'HM2'] = HM2
    data.loc[:,'AM2'] = AM2
    data.loc[:,'HM3'] = HM3
    data.loc[:,'AM3'] = AM3

    return data

#getPerformanceOfLast3Matches(training_data)
#print(training_data)

# -------create features---------
'''
getDistance(training_data,geometricData)

unifyDateFormat(training_data)
getMW(training_data,2008)
getDeltaTime(training_data)
getCumulativeGoalsDiff(training_data)
getAverageGD(training_data)
getPerformanceOfLast3Matches(training_data)

'''

# --------------- 删除中间数据 -----------------
# !!!【在notebook中不写成函数，直接写里面的代码】
def removeIntermediateData(data):   # or removeUnwantedData(data)
    data = data[data.MW > 3]
    
    data = removeInvalidData(data)

    return data

# training_data = removeIntermediateData(training_data)

# (--------------- Progress Summary -----------------)
# !!!【不必写成函数；重点是给一个总结，并且说明feature是28个，因为FTR是标签不是feature】
def printOutSummary(data):
    n_matches = data.shape[0]
    n_features = data.shape[1] - 1  # FTR is a label, not feature

    print("total number of matches: {}".format(n_matches))
    print("number of features: {}".format(n_features)) 
    print("home team win rate: {}".format(checkAverageWinRate(training_data,'H')))
    # print ("away team xxxxxxxx")
    # print ("draw xxxxxxxx")

#results of processed data:
#total number of matches = 3981
#number of features = 28  
#home team = 0.4654609394624466 ~ 0.4655; 
#away team = 0.2868625973373524 ~ 0.2869;
#draw = 0.24767646320020095 ~ 0.2477

# !!!【 in report: From these results, we can find that the processed data is still imbalanced. We chose to make it binary.】

# --------------- Data Transformation -----------------
# ********************************
#data = training_data.copy()
#data.drop(['Date','HomeTeam', 'AwayTeam', 'Referee','FTHG', 'FTAG', 'MW'],1, inplace=True)

# ********************************
# simplify to a binary problem, make the target be FTR == 'H'
def simplifyLabel(label):
    if label == 'H':
        return 'H'
    else:
        return 'NH'

#data['FTR'] = data.FTR.apply(simplifyLabel)
#data['HTR'] = data.HTR.apply(simplifyLabel)

# ********************************
# separate the training data into : feature set, label
#X_all = data.drop(['FTR'],1)
#Y_all = data['FTR']

# map the label into 0, 1
#Y_all = Y_all.map({'NH':0,'H':1})

# separate the columns by types: 
#categList = ["HTR", "HM1","AM1", "HM2","AM2", "HM3","AM3"]
#numList = list(set(X_all.columns.tolist()).difference(set(categList)))


# ********************************
# rescale data
def rescale(data, cols):
    for col in cols:
        max = data[col].max()
        min = data[col].min()
        data[col] = (data[col] - min) / (max - min)
    return data

#rescale(X_all,numList)   [not sure if needed to be the whole numList]

# ********************************
# standardization
from sklearn.preprocessing import scale
def standardize(data,cols):
    for col in cols:
        data[col] = scale(data[col])

#standardize(X_all, numList)

# ********************************
# transform categorical features
def transformCategoricalFeature(data,categoricalFeatureNames):
    # 把这些特征转换成字符串类型
    for col in categoricalFeatureNames:
        data[col] = data[col].astype('str')
    
    output = pd.DataFrame(index=data.index)

    for col_name, col_data in data.iteritems():
        if col_data.dtype == 'object':
            col_data = pd.get_dummies(col_data, prefix = col_name)
        output = output.join(col_data)
    
    return output

#X_all = transformCategoricalFeature(X_all, categList)


# --------------- Visualization -----------------
import matplotlib.pyplot as plt
import seaborn as sns
# ************************************
# plot all the features with Pearson correlation heatmap
def plotGraph(X_all, Y_all):

    train_data=pd.concat([X_all,Y_all],axis=1)

    colormap = plt.cm.RdBu
    plt.figure(figsize=(21,18))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(train_data.astype(float).corr(),linewidths=0.1,vmax=1.0,
                square=True, cmap=colormap, linecolor='white', annot=True)

# plotGraph(X_all, Y_all)
# !!!【in report: found that HAGD & HCGD, AAGD & ACGD are highly correlated, so drop HCGD, ACGD】
# X_all = X_all.drop(["HCGD","ACGD"], axis=1)

# *************************************
# plot the top 10 features related to FTR
def plotGraph2(X_all, Y_all):

    train_data=pd.concat([X_all,Y_all],axis=1)

    #FTR correlation matrix
    plt.figure(figsize=(14,12))
    k = 10 # number of variables for heatmap
    cols = abs(train_data.astype(float).corr()).nlargest(k, 'FTR')['FTR'].index
    cm = np.corrcoef(train_data[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()

#plotGraph2(X_all, Y_all)
# !!! 【in report: give a few comment of this graph】

# X_all = X_all["A", "B", "C", xxxx] 
# select the top 10 features according to the graph2, drop others

# -------------------- Classifiers ------------------------- 
'''
from sklearn.naive_bayes import GaussianNB
clf1 = GaussianNB()

from sklearn.linear_model import LogisticRegression
clf2 = LogisticRegression(solver='lbfgs', multi_class = 'multinomial')

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
clf3 = LDA()

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
clf4 = QDA()

from sklearn.tree import DecisionTreeClassifier
clf5 = DecisionTreeClassifier()

from sklearn.neural_network import MLPClassifier
clf6 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)

clfs = [clf1, clf2, clf3, clf4, clf5, clf6]
'''

# -------------------- Evaluation ------------------------- 
# ********************************
# split data
#X_train, X_test, y_train, y_test = train_test_split(X_all, Y_all, test_size = 50,random_state = 2,stratify = Y_all)

# ********************************
from time import time
from sklearn.metrics import f1_score
# train classifier
def train_classifier(clf, X_train, y_train):
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    print("time for training: {:.4f} sec".format(end - start))

# predict using the classifier
def predict_labels(clf, features, target):
    start = time()
    y_pred = clf.predict(features)
    end = time()
    print("time for prediction: {:.4f} sec".format(end - start))
    return f1_score(target, y_pred, pos_label=1), sum(target == y_pred) / float(len(y_pred))

# print out the performance of each classifer
def train_predict(clf, X_train, y_train, X_test, y_test):

    print("Classifier: {} [sample size: {}]".format(clf.__class__.__name__, len(X_train)))

    train_classifier(clf, X_train, y_train)

    # evaluate model on train set
    print("[on train set]")
    f1, acc = predict_labels(clf, X_train, y_train)
    print("F1 score: {:.4f} ".format(f1))
    print("accuracy: {:.4f}".format(acc))

    # evaluate model on test set
    print("[on test set]")
    f1, acc = predict_labels(clf, X_test, y_test)
    print("F1 score: {:.4f} ".format(f1))
    print("accuracy: {:.4f}".format(acc))

'''
for clf in clfs:
    train_predict(clf, X_train, y_train, X_test, y_test)
    print("\n")
'''
# [in report: xxx takes the shortest time for training; xxx has the highest accuracy; xxx [give comments to the result]]
# [in report: so we choose to adjust xxxx (the relatively best one among them) with hyperparameters]
# 【如果用ensemble method可行的话，就灵活调整这边的内容和顺序】

# ********************************
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
# adjust the model with hyperparameter      【如果只一个model的话可以不用写成函数】
def adjustClassifier(clf, f1_scorer, param, X_train, y_train):

    grid_obj = GridSearchCV(clf,scoring=f1_scorer,param_grid=param,cv=5)
    grid_obj = grid_obj.fit(X_train,y_train)

    clf = grid_obj.best_estimator_

    return clf

# clf = LogisticRegression(xxxxx)
# f1_scorer = make_scorer(xxxxxxxxx)
# parameters = {xxxxx}
# clf = adjustClassifier(clf, f1_scorer, parameters, X_train, y_train)

'''
Parameter Settings Plans of Each Classifier:
# Logistic Regression
clf = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial')
f1_scorer = make_scorer(f1_score, average = 'weighted')
parameters = { 
              'C' :[1.0, 100.0, 1000.0],
              'max_iter':[100,200,300, 400, 500],
              'intercept_scaling':[0.1, 0.5, 1.0]
             }

# GaussianNB
clf = GaussianNB()
f1_scorer = make_scorer(f1_score, average = 'weighted')
parameters = { 
              'var_smoothing': [1e-09, 1e-07, 1e-05, 1e-11, 1e-13]
             }

#LDA
clf = LDA()
f1_scorer = make_scorer(f1_score, average = 'weighted')
parameters = { 
              'solver': ['svd', 'lsqr', 'eigen'],
              'tol': [ 0.001, 0.0001, 0.00001]
             }

#QDA
clf = QDA()
f1_scorer = make_scorer(f1_score, average = 'weighted')
parameters = { 
              'reg_param': [0, 0.1, 0.01, 0.001],
              'tol': [0.001, 0.0001, 0.00001]
             }

#Decision Tree
clf = DecisionTreeClassifier()
f1_scorer = make_scorer(f1_score, average = 'weighted')
parameters = { 
              'solver': ['svd', 'lsqr', 'eigen'],
              'tol': [0.01, 0.001, 0.0001, 0.00001]
             }

#Neural Network
clf = MLPClassifier(solver='lbfgs', alpha=1e-5
                    , random_state=1)
f1_scorer = make_scorer(f1_score, average = 'weighted')
parameters = { 
              'alpha': [ 1e-03, 1e-05, 1e-07],
              'hidden_layer_sizes':[ (5,), (10,), (15,)],
              'learning_rate_init':[0.01, 0.001, 0.0001],
             }

'''

# **************************
# ensemble learning     【to choose the best model?】    
'''
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve

#Random forrest
kfold_rf = model_selection.KFold(n_splits=10, random_state=10)
model_rf = RandomForestClassifier(n_estimators=100, max_features=5)
results_rf = model_selection.cross_val_score(model_rf, X_train, y_train, cv=kfold_rf)
print(results_rf.mean())
# 0.76776514272453

#adaBoost
from sklearn.ensemble import AdaBoostClassifier
kfold_ada = model_selection.KFold(n_splits=10, random_state=10)
model_ada = AdaBoostClassifier(n_estimators=30, random_state=10)
results_ada = model_selection.cross_val_score(model_ada,X_train, y_train, cv=kfold_ada)
print(results_ada.mean())
# 0.7742270699569377

#Gradient Boost
from sklearn.ensemble import GradientBoostingClassifier
kfold_sgb = model_selection.KFold(n_splits=10, random_state=10)
model_sgb = GradientBoostingClassifier(n_estimators=100, random_state=10)
results_sgb = model_selection.cross_val_score(model_sgb, X_train, y_train, cv=kfold_sgb)
print(results_sgb.mean())
# 0.7727959567829606

#voting estimator
kfold_vc = model_selection.KFold(n_splits=10, random_state=10)
 
estimators = []
mod_lr = GaussianNB()
estimators.append(('Gaussian', mod_lr))
mod_dt = DecisionTreeClassifier()
estimators.append(('cart', mod_dt))
mod_sv = SVC(gamma = 'scale')
estimators.append(('LDA', mod_sv))
ensemble = VotingClassifier(estimators, voting = 'hard', weights=[1,1,1])

results_vc = model_selection.cross_val_score(ensemble, X_train, y_train ,cv=kfold_vc)

print(results_vc.mean())
# 0.7745867821871535
'''

# ------------------- Derive Features of Test Sample -------------
# ***********************
# read test data
#url = 'https://raw.githubusercontent.com/Yun5141/comp0036/master/epl-test.csv'
#X_sample = pd.read_csv(url)

# ************************
# derive features:
# unifyDate(X_sample)
# getDistance(X_sample)
# xxxxxx
# xxxxxx

# -------------------- Results ------------------------- 

# 通过前面的方法预测出来的概率最高的类别即判断结果  [not sure]
def labelClassifier(H_rate, A_rate, D_rate):
    pass

# -------------------- Final Prediction ------------------------- 
#train_classifier(clf1,X_train,y_train)     # train the classifer
#sample1 = X_test.sample(n=1, random_state=1)
#y_pred = clf1.predict(sample1)
#y_pred  # 1 means home team wins; 0 means away team wins or draw