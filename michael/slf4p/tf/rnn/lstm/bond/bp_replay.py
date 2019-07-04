'''
Created on 2019年7月4日

@author: ch
'''
import pandas as pd
import pickle
from michael.slf4p.tf.rnn.lstm.bond.normalizeUtil import get_data

# namelist = ["Michael", "Wendy", "Vicky", "Sam", "George", "Rose"]
namelist = ["Michael"]
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
    print(name, ":expected value\tpredict value")
    for i in range(len(val_y)):
        print(str(val_y[i]) + "," + str(result_proba[i][1]))
