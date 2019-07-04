'''
Created on 2019年6月26日

@author: ch
'''
import pandas as pd
import pickle
from keras.models import load_model
from michael.slf4p.tf.rnn.lstm.bond.normalizeUtil import get_data
from michael.slf4p.tf.rnn.lstm.bond.normalizeUtil import get_XY

# namelist = ["Michael", "Wendy", "Vicky", "Sam", "George", "Rose"]
# Michael:1
# Wendy: 0.7 - 0.94 - 0.96
# Vicky: 0.818 - 0.946 - 0.98
# Sam: 0.816 - 0.988 - 0.998
# George: 0.966 - 0.988 - 0.99
# Rose: 1
namelist = ["Michael"]
for name in namelist:
    model = load_model(filepath='/Users/ch/git/tf/resources/model/' + name + '-bond.md')
    model.load_weights(filepath='/Users/ch/git/tf/resources/model/' + name + '-bond_weights.md')
    minmaxFile = '/Users/ch/git/tf/resources/minmax/' + name + '-minmax.pk'
    print("current minmax model file:", minmaxFile)
    with open(minmaxFile, 'rb') as fid:
        scaler = pickle.load(fid)
    # integer encode direction
    val_dataset = pd.read_csv('File:/Users/ch/git/tf/resources/data/'+ name + '-bid-seq-test.csv', header=0, index_col=0)
    val_values_X, val_y = get_data(val_dataset, scaler)
    # split into input and outputs
    val_X = get_XY(val_values_X)
#     print(type(val_X), val_X)
    actual_y = model.predict(val_X)
    print(name, ":expected value\tpredict value")
    for i in range(len(val_y)):
        print(str(val_y[i]) + "," + str(actual_y[i]).replace("[", "").replace("]", ""))
