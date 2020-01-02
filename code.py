# ------------------- import packages -----------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from os import chdir

# !pip install sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# may need to import sklearn.lda.LDA and sklearn.qda.QDA instead
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

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


# ------------------ Data Pre-Processing -------------------

# cleaning data (去掉不要的)    [not sure]
def dataCleaning(raw_data):
  raw_data.drop('Referee', axis = 1, inplace = True)

# ********************************************
# unify the different date formats and convert the type from str to timestamp   [done]
def unifyDateFormat(data):

    if not isinstance(data.Date[0],str):
        return

    newDate = []
    for _, matchInfo in data.iterrows():
        if len(matchInfo.Date) == 8 :
            newDate.append( pd.to_datetime(matchInfo.Date, format="%d/%m/%y" ))
        elif len(matchInfo.Date) == 10 :
            newDate.append(  pd.to_datetime(matchInfo.Date, format="%d/%m/%Y" ))
    
    data['Date'] = pd.Series(newDate).values

unifyDateFormat(training_data)

# ------------------ Inital Data Exploration -------------------
# ********************************************
# !!! 【report中: 第一步先检查有无空值】
# will print out the number of null value in each column
training_data.isnull().sum()
#result: there is no empty value at the initial stage     

# ********************************************
# to see the number of matches each year / season
def separateData(data):
    dataframe_collection = {}

    for year in range(2008, 2020):
        dataframe_collection[year] = data[(data.Date > dt.datetime(year,8,1,0,0) ) & (data.Date < dt.datetime(year+1, 6, 1,0,0))]

    return dataframe_collection

data = separateData(training_data)
for key in data.keys():
    print("\n" +"="*40)
    print(key)
    print("-"*40)
    print(data[key])

#result: 380 rows * 11 dataframes + 170 rows * 1 dataframes = 4350 rows

# ********************************************
# !!! 【在数据正式处理前后各用一次这个函数，即两次的data exploration section】
def checkAverageWinRate(data, resultWinner):

    if resultWinner not in ['H', 'A', 'D']:
        raise Exception('The second argument should only take values within [“H”,“A”,“D”]')

    predictions = 0
    for _, matchInfo in data.iterrows():
        
        if matchInfo['FTR'] == resultWinner:
          predictions += 1

    return predictions / len(data)

#prediction = checkAverageWinRate(training_data, 'H')

#results of raw data (ie, when nothing applied to the training data): 
#home team = 0.4606896551724138 ~ 0.461; 
#away team = 0.2910344827586207 ~ 0.291;
#draw = 0.2482758620689655 ～ 0.248

#results of processed data:
#home team = ; 
#away team = ;
#draw = 

# ------------------- Feature Construction ------------------——————————
#*******************************
# get the distance needed to travel for the away team   [done] 
def getDistance(training_data, geometricData):

  Teams = training_data.HomeTeam
  geometricData = geometricData.loc[geometricData['Team'].isin(Teams)]

  array = []
  for x in training_data.iterrows():
    home_lat = (geometricData.loc[geometricData['Team'] == x[1].HomeTeam]).Latitude
    home_long = (geometricData.loc[geometricData['Team'] == x[1].HomeTeam]).Longitude
    home_location = (np.float32(home_lat), np.float32(home_long))
    away_lat = (geometricData.loc[geometricData['Team'] == x[1].AwayTeam]).Latitude
    away_long = (geometricData.loc[geometricData['Team'] == x[1].AwayTeam]).Longitude
    away_location = (np.float32(away_lat), np.float32(away_long))
    array.append(great_circle(home_location, away_location).km)

  DIS = pd.Series(array)
  training_data.loc[:,'DIS'] = DIS.values

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

    for name in data.groupby('HomeTeam').mean().T.columns:
        teams[name] = []

    # for each match
    for i in range(len(data)):
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

# --------------- 删除中间数据 -----------------
# !!!【在notebook中不写成函数，直接写里面的代码】
def removeIntermediateData(data):   # or removeUnwantedData(data)
    data = data[data.MW > 3]
    
    print(data.isnull().sum())
    # if there are empty values
    # data.dropna(axis=0, how='any')

    return data

# --------------- Second Data Exploration -----------------

# data visualization
def plotGraph(data):
    pass

# after viewing the result of the visualization, remove the attributes that is too related to the result
# !!!【在notebook中不写成函数，直接写里面的代码】
def selectFeatures(data):
    pass

# -------------------- Model (by Yanke)------------------------- 
'''
- haven't re-organized
- only involves a single model, logistic regression.
- Yanke is now working on model combination (as suggested in the marking guidelines)
'''
# ---------------
# training_data2 = pd.read_csv('/content/training_data_with_distance.csv')
# a = []
# for i in np.arange(20, 73, 1).tolist():
#   a.append(i)

# training_data = training_data.drop(training_data.columns[a], axis = 1)
training_data = training_data.drop(['Date'],1)
def only_hw(string):
    if string == 'H':
        return 1
    if string == 'A':
        return 0
    else:
        return 2

training_data.FTR = training_data.FTR.apply(only_hw)
training_data.HTR = training_data.HTR.apply(only_hw)
referee = dict(zip(list(dict.fromkeys(training_data.Referee)), np.arange(0, len(dict.fromkeys(training_data.Referee)), 1).tolist()))
result = dict(zip(list(dict.fromkeys(training_data.HomeTeam)), np.arange(0, 36, 1).tolist()))

def referee_to_num(string):
    if string in referee:
        return referee[string]
def team_to_num(string):
    if string in result:
        return result[string]
training_data.HomeTeam = training_data.HomeTeam.apply(team_to_num)   
training_data.AwayTeam = training_data.AwayTeam.apply(team_to_num) 
training_data.Referee = training_data.Referee.apply(referee_to_num) 

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)
clean_dataset(training_data)

X_all = training_data.drop(['FTR'],1).drop(['FTHG'],1).drop(['FTAG'],1)
y_all = training_data['FTR'] 


X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,test_size = 0.3,random_state = 2,stratify = y_all)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_all, y_all,test_size = 0.1,random_state = 2,stratify = y_all)

print(X_train, X_test)

# ------- plot ------------
corr = X_all.corr()

from seaborn import heatmap
heatmap(corr)

plt.show()

# -------- regression ---------
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_lr = lr.predict(X_test)
y_lr_prob = lr.predict_proba(X_test)[:, -1]  # probability estimates of the positive class


# create a dictionary variable with keys being algorithm names and values being classification accuracy

accuracy = accuracy_score(y_test, y_lr)
    

print(accuracy)