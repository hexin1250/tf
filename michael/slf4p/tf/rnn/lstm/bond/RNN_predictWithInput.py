'''
Created on 2019年7月1日

@author: ch
'''
import pandas as pd
import numpy as np
import sys
import datetime
from sklearn.preprocessing import LabelEncoder
import pickle
from keras.models import load_model
from michael.slf4p.tf.rnn.lstm.bond.normalizeUtil import get_data
from michael.slf4p.tf.rnn.lstm.bond.normalizeUtil import get_XY
from michael.slf4p.tf.rnn.lstm.bond.normalizeUtil import consist_data

# namelist = ["George", "Michael", "Sam", "Wendy", "Rose", "Vicky"]
# args = sys.argv[1:]
yield_v = 0.09957762
price_v = 101.77377358
rating = 6
args = [yield_v * price_v, rating, price_v, yield_v]
# args = [2.17947768,4,100.60311253,0.02166412]
namelist = ["Michael"]
for i in range(len(namelist)):
    name = namelist[i]
    model = load_model(filepath='/Users/ch/git/tf/resources/rnn/model/' + name + '-bond.md')
    model.load_weights(filepath='/Users/ch/git/tf/resources/rnn/model/' + name + '-bond_weights.md')
    minmaxFile = '/Users/ch/git/tf/resources/rnn/minmax/' + name + '-minmax.pk'
    with open(minmaxFile, 'rb') as fid:
        scaler = pickle.load(fid)
    # integer encode direction
    encoder = LabelEncoder()
    dataset = consist_data(sys_args=args, order=i)
    val_values_X, val_y = get_data(dataset, scaler)
    # split into input and outputs
    val_X = get_XY(val_values_X)
    actual_y = model.predict(val_X)
    print(str(actual_y[0]).replace("[", "").replace("]", ""))
#     print(name, "predict value", str(actual_y[0]).replace("[", "").replace("]", ""))
