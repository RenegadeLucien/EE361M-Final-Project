# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 21:32:03 2016

@author: Michael
"""

import pandas as pd
import numpy as np
"from ml_metrics import average_precision"
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
print("training data loaded")
dest = pd.read_csv('destinations_mod.csv')
print("dest data loaded")
test = pd.read_csv('test.csv', parse_dates=['date_time', 'srch_co'])
print("test data loaded")
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
data = data.drop('site_name', 1)
data = data.drop('posa_continent', 1)
data = data.drop('is_mobile', 1)
data = data.drop('is_package', 1)
data = data.drop('srch_adults_cnt', 1)
data = data.drop('srch_children_cnt', 1)
data = data.drop('srch_rm_cnt', 1)
data = data.drop('srch_destination_type_id', 1)
data = data.drop('hotel_continent', 1)
data_y = data["hotel_cluster"]
data = data.drop('hotel_cluster', 1)
test = test.drop('site_name', 1)
test = test.drop('posa_continent', 1)
test = test.drop('is_mobile', 1)
test = test.drop('is_package', 1)
test = test.drop('srch_adults_cnt', 1)
test = test.drop('srch_children_cnt', 1)
test = test.drop('srch_rm_cnt', 1)
test = test.drop('srch_destination_type_id', 1)
test = test.drop('hotel_continent', 1)
test['srch_ci'].replace(to_replace='nan', value=pd.to_datetime('1/1/2017'), inplace=True)
test.set_value(312920, 'srch_ci', pd.to_datetime('1/21/2016'))
test['srch_ci'] = pd.to_datetime(test['srch_ci'], errors='coerce')
test['srch_ci'].fillna(value=pd.to_datetime('1/1/2017'), inplace=True)
test['srch_co'].fillna(value=pd.to_datetime('1/8/2017'), inplace=True)
print("pp part 1 complete")
                
def addFeatures(df_input):
    df_input['date_time_month'] = df_input['date_time'].dt.month
    df_input['date_time_day'] = df_input['date_time'].dt.day
    df_input['date_time_hour'] = df_input['date_time'].dt.hour
    df_input['srch_ci_month'] = df_input['srch_ci'].dt.month
    df_input['srch_ci_dayofweek'] = df_input['srch_ci'].dt.dayofweek
    df_input['srch_ci_day'] = df_input['srch_ci'].dt.day
    df_input['srch_co_month'] = df_input['srch_co'].dt.month
    print("new features added")
    df_input = df_input.drop('date_time', 1)
    df_input = df_input.drop('srch_ci', 1)
    df_input = df_input.drop('srch_co', 1)
    df_input = pd.merge(df_input, dest, on='srch_destination_id', how='left')
    df_input['orig_destination_distance'].fillna(1850, inplace=True)
    df_input.fillna(-2.18, inplace=True)
    df_input = df_input.drop('srch_destination_id', 1)
    print("pp part 2 complete")
    return df_input
    
data = addFeatures(data)
test = addFeatures(test)


decisionTree = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=20, 
                                       min_samples_split=2, min_samples_leaf=1, 
                                       min_weight_fraction_leaf=0.0, max_features=None, 
                                       random_state=None, max_leaf_nodes=None, 
                                       min_impurity_split=1e-07, class_weight=None, presort=False)

decisionTree.fit(data, data_y)
probs = decisionTree.predict_proba(test)
pred = [];
indices = [];
for i in xrange(len(test)):
    pred.append(" ".join(map(str, list(reversed(probs[i].argsort()[-5:])))))
    indices.append(i)  

df = pd.DataFrame({"id" : indices, "hotel_cluster" : pred})
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]
df.to_csv("result2.csv", index=False)