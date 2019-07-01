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
# Wendy: 0.912 - 0.83 - 0.962
# Vicky: 0.492 - 0.64 - 0.572
# Sam: 0.612 - 0.556
# George: 0.96
# Rose: 1
namelist = ["George"]
for name in namelist:
    model = load_model(filepath='/Users/ch/git/tf/resources/model/' + name + '-bond.md')
    model.load_weights(filepath='/Users/ch/git/tf/resources/model/' + name + '-bond_weights.md')
    with open('/Users/ch/git/tf/resources/minmax/' + name + '-minmax.pk', 'rb') as fid:
        scaler = pickle.load(fid)
    # integer encode direction
    val_dataset = pd.read_csv('File:/Users/ch/git/tf/resources/data/'+ name + '-bid-seq-test.csv', header=0, index_col=0)
    print(val_dataset)
    val_values = get_data(val_dataset, scaler)
    # split into input and outputs
    val_X, val_y = get_XY(val_values)
    actual_y = model.predict(val_X)
    print(name, ":expected value\tpredict value")
    for i in range(len(val_y)):
        print(str(val_y[i]) + "," + str(actual_y[i]).replace("[", "").replace("]", ""))
