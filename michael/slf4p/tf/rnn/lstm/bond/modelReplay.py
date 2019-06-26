'''
Created on 2019年6月26日

@author: ch
'''
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from michael.slf4p.tf.rnn.lstm.bond.normalizeUtil import get_data
from michael.slf4p.tf.rnn.lstm.bond.normalizeUtil import get_XY

name = "Michael"
model = load_model(filepath='/Users/ch/git/tf/resources/model/' + name + '-bond.md')
model.load_weights(filepath='/Users/ch/git/tf/resources/model/' + name + '-bond_weights.md')
scaler = MinMaxScaler(feature_range=(0,1))
# integer encode direction
encoder = LabelEncoder()
val_dataset = pd.read_csv('File:/Users/ch/git/tf/resources/data/'+ name + '-bid-seq-test.csv', header=0, index_col=0)
val_values = get_data(val_dataset, scaler, encoder)
# split into input and outputs
val_X, val_y = get_XY(val_values)
actual_y = model.predict(val_X)
print(name, ":expected value\tpredict value")
for i in range(len(val_y)):
    print(val_y[i], actual_y[i])
