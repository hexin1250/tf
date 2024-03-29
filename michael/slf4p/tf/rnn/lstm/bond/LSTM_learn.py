'''
Created on 2019年6月26日

@author: ch
'''
import pandas as pd
import keras
import pickle
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import load_model
from michael.slf4p.tf.rnn.lstm.bond.normalizeUtil import get_data
from michael.slf4p.tf.rnn.lstm.bond.normalizeUtil import get_XY
from multiprocessing import active_children

times=1000
namelist = ["George", "Michael", "Sam", "Wendy", "Rose", "Vicky"]
# namelist = ["George"]
for name in namelist:
    dataset = pd.read_csv('File:/Users/ch/git/tf/resources/data/' + name + '-bid-seq-train.csv', header=0, index_col=0)
    scaler = MinMaxScaler(feature_range=(0,1))
    values_train_X, train_y = get_data(dataset, scaler, fit=True)
    train_X = get_XY(values_train_X)
    with open('/Users/ch/git/tf/resources/lstm/minmax/' + name + '-minmax.pk', 'wb') as fid:
        pickle.dump(scaler, fid)
    
    validation_ds = pd.read_csv('/Users/ch/git/tf/resources/data/' + name + '-bid-seq-val.csv', header=0, index_col=0)
    values_test_X, test_y = get_data(validation_ds, scaler)
    test_X = get_XY(values_test_X)
     
#     model = keras.models.Sequential()
#     model.add(keras.layers.LSTM(10, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
#     model.add(keras.layers.LSTM(10, return_sequences=True))
#     model.add(keras.layers.LSTM(10))
#     model.add(keras.layers.Dense(1, activation='sigmoid'))
#     model.compile(loss='mse', optimizer='nadam', metrics=['accuracy'])

    model = load_model(filepath='/Users/ch/git/tf/resources/lstm/model/' + name + '-bond.md')
    model.load_weights(filepath='/Users/ch/git/tf/resources/lstm/model/' + name + '-bond_weights.md')
    print("current train client:", name)
#     batch_size = int(len(train_y) / 3)
    batch_size = 100
    history = model.fit(train_X, train_y, epochs=times, batch_size=batch_size, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    
    # save model
    model.save(filepath='/Users/ch/git/tf/resources/lstm/model/' + name + '-bond_weights.md')
    model.save(filepath='/Users/ch/git/tf/resources/lstm/model/' + name + '-bond.md')
    # val_dataset = pd.read_csv('File:/Users/ch/git/tf/test-seq.csv', header=0, index_col=0)
#     val_dataset = pd.read_csv('File:/Users/ch/git/tf/resources/data/' + name + '-bid-seq-test.csv', header=0, index_col=0)
#     val_values = get_data(val_dataset, scaler)
#     # split into input and outputs
#     val_X, val_y = get_XY(val_values)
#     actual_y = model.predict(val_X)
#     print(name, ":expected value\tpredict value")
#     for i in range(len(val_y)):
#         print(val_y[i], actual_y[i])