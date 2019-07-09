'''
Created on 2019年7月4日

@author: ch
'''
import pandas as pd
import pickle
from michael.slf4p.tf.rnn.lstm.bond.normalizeUtil import get_data

namelist = ["George", "Michael", "Sam", "Wendy", "Rose", "Vicky"]
# namelist = ["George"]
for name in namelist:
    val_dataset = pd.read_csv('File:/Users/ch/git/tf/resources/data/'+ name + '-bid-seq-test.csv', header=0, index_col=0)
    
    minmaxFile = '/Users/ch/git/tf/resources/bp/minmax/' + name + '-minmax.pk'
    with open(minmaxFile, 'rb') as fid:
        scaler = pickle.load(fid)
    val_X, val_y = get_data(val_dataset, scaler)
    modelFileName = '/Users/ch/git/tf/resources/bp/model/' + name + '.model'
    with open(modelFileName, 'rb') as fid:
        clf = pickle.load(fid)
    result = clf.predict(val_X)
    result_proba = clf.predict_proba(val_X)
#     for i in range(len(val_y)):
#         print(val_y[i], result_proba[i])
    
    length = len(val_y)
    arr = [0, 0, 0, 0, 0]
    
    print(name, ":expected value\tpredict value")
    for i in range(length):
        new_value = result_proba[i][1]
        if new_value > 1:
            new_value = 1
        elif new_value < 0:
            new_value = 0
        value = abs(new_value - val_y[i])
        if value < 0.1:
            arr[0] += 1
            arr[1] += 1
            arr[2] += 1
            arr[3] += 1
            arr[4] += 1
        elif value < 0.2:
            arr[1] += 1
            arr[2] += 1
            arr[3] += 1
            arr[4] += 1
        elif value < 0.3:
            arr[2] += 1
            arr[3] += 1
            arr[4] += 1
        elif value < 0.4:
            arr[3] += 1
            arr[4] += 1
        elif value < 0.5:
            arr[4] += 1
    print([x / length for x in arr])
