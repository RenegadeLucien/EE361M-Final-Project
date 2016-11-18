# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 21:32:03 2016

@author: Michael
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

train0 = pd.read_csv('train_0.csv', header=None, skiprows=1, parse_dates=[0,11,12])
train1 = pd.read_csv('train_1.csv', header=None, parse_dates=[0,11,12])
train2 = pd.read_csv('train_2.csv', header=None, parse_dates=[0,11,12])
train3 = pd.read_csv('train_3.csv', header=None, parse_dates=[0,11,12])
train4 = pd.read_csv('train_4.csv', header=None, parse_dates=[0,11,12])
train5 = pd.read_csv('train_5.csv', header=None, parse_dates=[0,11,12])
train6 = pd.read_csv('train_6.csv', header=None, parse_dates=[0,11,12])
train7 = pd.read_csv('train_7.csv', header=None, parse_dates=[0,11,12])
train8 = pd.read_csv('train_8.csv', header=None, parse_dates=[0,11,12])
train9 = pd.read_csv('train_9.csv', header=None, parse_dates=[0,11,12])
dest = pd.read_csv('destinations.csv')
test = pd.read_csv('test.csv', parse_dates=['date_time', 'srch_ci', 'srch_co'])
test = test.drop('id', 1)

data = pd.concat([train0, train1, train2, train3, train4, train5, train6, train7, train8, train9])
data.columns = ["date_time", "site_name", "posa_continent", "user_location_country",
                "user_location_region", "user_location_city", "orig_destination_distance",
                "user_id", "is_mobile", "is_package", "channel", "srch_ci", "srch_co",
                "srch_adults_cnt", "srch_children_cnt", "srch_rm_cnt", "srch_destination_id",
                "srch_destination_type_id", "hotel_continent", "hotel_country",
                "hotel_market", "is_booking", "cnt", "hotel_cluster"]    
            
data = data.drop('cnt', 1)
data = data.drop('is_booking', 1)
data_y = data["hotel_cluster"]
data = data.drop('hotel_cluster', 1)
test['srch_ci'].replace(to_replace='nan', value=pd.to_datetime('1/1/2017'), inplace=True)
test.loc['srch_ci'] = pd.to_datetime(test['srch_ci'], errors='coerce')
test['srch_ci'].fillna(value=pd.to_datetime('1/1/2017'), inplace=True)
test['srch_co'].fillna(value=pd.to_datetime('1/8/2017'), inplace=True)
                
def addFeatures(df_input):
    df_output = df_input.copy()
    df_output['date_time_month'] = df_output['date_time'].dt.month
    df_output['date_time_year'] = df_output['date_time'].dt.year
    df_output['date_time_dayofweek'] = df_output['date_time'].dt.dayofweek
    df_output['date_time_day'] = df_output['date_time'].dt.day
    df_output['date_time_hour'] = df_output['date_time'].dt.hour
    df_output['srch_ci_month'] = df_output['srch_ci'].dt.month
    df_output['srch_ci_year'] = df_output['srch_ci'].dt.year
    df_output['srch_ci_dayofweek'] = df_output['srch_ci'].dt.dayofweek
    df_output['srch_ci_day'] = df_output['srch_ci'].dt.day
    df_output['srch_co_month'] = df_output['srch_co'].dt.month
    df_output['srch_co_year'] = df_output['date_time'].dt.year
    df_output['srch_co_dayofweek'] = df_output['date_time'].dt.dayofweek
    df_output['srch_co_day'] = df_output['srch_co'].dt.day
    df_output = df_output.drop('date_time', 1)
    df_output = df_output.drop('srch_ci', 1)
    df_output = df_output.drop('srch_co', 1)
    df_output = pd.merge(df_output, dest, on='srch_destination_id', how='left')
    df_output['orig_destination_distance'].fillna(1850, inplace=True)
    df_output.fillna(-2.18, inplace=True)
    cols1 = [14,15,31,32,34,35,37,38,42,43,45,48,49,51,52,53,56,57,58,60,61,62,63,64,
         66,69,70,72,73,74,76,77,78,80,81,82,83,84,86,87,88,89,90,91,92,94,95,
         97,98,99,100,102,103,107,108,110,111,112,113,114,115,116,117,119,124,125,
         127,128,129,131,132,133,134,135,136,137,139,140,141,142,143,144,146,147,148,149,
         150,153,154,156,157,159,160,161,162,163,165,166,168,170,172,173,174,175,176,177,178,179]
    df_output = df_output.drop(df_output.columns[cols1], 1)
    cols2 = [1,7,12,13,17,26,36,28,39,40,42,44,45,47,48,51,53,54,56,57,60,61,65,66,68,71]
    df_output = df_output.drop(df_output.columns[cols2], 1)
    cols3 = [6,30,31,43]
    df_output = df_output.drop(df_output.columns[cols3], 1)
    cols4 = [8,24,30,34,41]
    df_output = df_output.drop(df_output.columns[cols4], 1)
    cols5 = [0,11,15,19]
    df_output = df_output.drop(df_output.columns[cols5], 1)
    df_output = df_output.drop(df_output.columns[6], 1)
    return df_output
    
data = addFeatures(data)
test = addFeatures(test)




decisionTree = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=20, 
                                       min_samples_split=2, min_samples_leaf=1, 
                                       min_weight_fraction_leaf=0.0, max_features=None, 
                                       random_state=None, max_leaf_nodes=None, 
                                       min_impurity_split=1e-07, class_weight=None, presort=False)

decisionTree.fit(data, data_y)
print decisionTree.predict_proba(test)
