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

def convert_to_dataframe(arr):
    std_arr = ['', 'result','name','cpn','rate','price','yield']
    new_arr = []
    new_arr.append(std_arr)
    new_arr.append(arr)
    data = np.array(new_arr)
    return pd.DataFrame(data=data[1:, 1:], index=data[1:, 0], columns=data[0, 1:])

def consist_data(sys_args, order):
    arr = []
    arr.append(datetime.datetime.now())
#     arr.append(datetime.datetime.strptime('2016 07 16 01', '%Y %m %d %H'))
    arr.append(1)
    arr.append(order)
    for item in sys_args:
        arr.append(item)
    return convert_to_dataframe(arr)

namelist = ["Michael", "Wendy", "Vicky", "Sam", "George", "Rose"]
# args = sys.argv[1:]
args = [2.25438205,6,100.07637840000001,0.02252662]
# args = [2.17947768,4,100.60311253,0.02166412]
# namelist = ["Wendy"]
for i in range(len(namelist)):
    name = namelist[i]
    model = load_model(filepath='/Users/ch/git/tf/resources/model/' + name + '-bond.md')
    model.load_weights(filepath='/Users/ch/git/tf/resources/model/' + name + '-bond_weights.md')
    minmaxFile = '/Users/ch/git/tf/resources/minmax/' + name + '-minmax.pk'
    print("current minmax model file:", minmaxFile)
    with open(minmaxFile, 'rb') as fid:
        scaler = pickle.load(fid)
    # integer encode direction
    encoder = LabelEncoder()
    dataset = consist_data(sys_args=args, order=i)
    val_values_X, val_y = get_data(dataset, scaler)
    # split into input and outputs
    val_X = get_XY(val_values_X)
    print(val_X)
    actual_y = model.predict(val_X)
    print(name, "predict value", str(actual_y[0]).replace("[", "").replace("]", ""))
