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

data = pd.concat([train0, train1, train2, train3, train4, train5, train6, train7, train8, train9])
data.columns = ["date_time", "site_name", "posa_continent", "user_location_country",
                "user_location_region", "user_location_city", "orig_destination_distance",
                "user_id", "is_mobile", "is_package", "channel", "srch_ci", "srch_co",
                "srch_adults_cnt", "srch_children_ct", "srch_rm_cnt", "srch_destination_id",
                "srch_destination_type_id", "hotel_continent", "hotel_country",
                "hotel_market", "is_booking", "cnt", "hotel_cluster"]    
            
data_y = data["hotel_cluster"]
data = data.drop('hotel_cluster', 1)
                
def addFeatures(df_input):
    df_output = df_input.copy()
    df_output['date_time_month'] = df_input['date_time'].dt.month
    return df_output
    
data = addFeatures(data)
data = data.drop('date_time', 1)
data = data.drop('srch_ci', 1)
data = data.drop('srch_co', 1)
data.fillna(0, inplace=True)

data = data.drop('hotel_continent', 1)
data = data.drop('hotel_country', 1)
data = data.drop('srch_rm_cnt', 1)
data = data.drop('channel', 1)
data = data.drop('is_mobile', 1)
data = data.drop('posa_continent', 1)
data = data.drop('srch_adults_cnt', 1)
data = data.drop('site_name', 1)
data = data.drop('is_package', 1)
data = data.drop('srch_children_ct', 1)
data = data.drop('user_location_country', 1)
data = data.drop('user_location_region', 1)
data = data.drop('user_location_city', 1)

decisionTree = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=10, 
                                       min_samples_split=2, min_samples_leaf=1, 
                                       min_weight_fraction_leaf=0.0, max_features=None, 
                                       random_state=None, max_leaf_nodes=None, 
                                       min_impurity_split=1e-07, class_weight=None, presort=False)

decisionTree.fit(data, data_y)
print decisionTree.score(data, data_y)
print decisionTree.feature_importances_
print list(data.columns.values)
