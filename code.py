# ------------------- import packages -----------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from os import chdir
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# you may need to import sklearn.lda.LDA and sklearn.qda.QDA instead
# depending on which version you have installed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve

import io
from geopy.distance import geodesic 
from geopy.distance import great_circle 

import datetime as dt


# ------------------- import data -------------------------
url="https://raw.githubusercontent.com/Yun5141/comp0036/master/elp_up-to-date_data.csv"
training_data=pd.read_csv(url)
url = "https://raw.githubusercontent.com/Yun5141/comp0036/master/stadiums-with-GPS-coordinates.csv"
geometricData = pd.read_csv(url)


# ------------------ Data Pre-Processing -------------------

# cleaning data (去掉不要的)
def dataCleaning(raw_data):
  raw_data.drop('Referee', axis = 1, inplace = True)

# ********************************************
# 把数据的赛季分开 [not sure if necessary]
def separateData(data):
    data.Date = pd.to_datetime(data.Date)

    dataframe_collection = {}

    for year in range(2008, 2019):
        dataframe_collection[year] = data[(data.Date > dt.datetime(year,8,1,0,0) ) & (data.Date < dt.datetime(year+1, 6, 1,0,0))]

    return dataframe_collection
        
data = separateData(training_data)
for key in data.keys():
    print("\n" +"="*40)
    print(key)
    print("-"*40)
    print(data[key])
# Bug：会漏数据，比如2008年表中最后数据的时间应为2009-05-24，但结果却是2009-05-12；
#       如果单独判定2009-05-24在不在范围内，结果又是正确的:
# print((training_data.Date[379] > dt.datetime(2008,8,1)) & (training_data.Date[379] < dt.datetime(2009,6,1)) )

# ------------------- Feature Construction ------------------——————————

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
  training_data.loc[:,'DIS'] = DIS

#getDistance(training_data, geometricData) 
#print(training_data)

# get match week
def getMW(data):
    pass

# get the interval time between two matches for each team   [by Yi]
def getIntervalTime(data):
    pass

# average away team win rate [necessary？]
def getAWR(data):
    pass

# 计算每个队周累计净胜球数量（goal_diff)
def getGoalsDiff(data):
    pass

# 统计每支队伍最近三场比赛的表现
def getPerformanceOfLast3Matches(data):
    pass


# --------------- Data Exploration -----------------

# data visualization
def plotGraph(data):
    pass

# after viewing the result of the visualization, 去掉与结果过于相关的数据
# 在notebook中不写成函数，直接写里面的代码
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