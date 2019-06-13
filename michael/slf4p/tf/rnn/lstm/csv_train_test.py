'''
Created on 2019年6月13日

@author: ch
'''
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from codecs import ignore_errors
from jupyter_core.troubleshoot import get_data

train_date_url = "File:/Users/ch/eclipse-workspace/neural-rnn/src/main/resources/data/train.data"
train_file_path = tf.keras.utils.get_file("train.csv", train_date_url)

np.set_printoptions(precision=3, suppress=True)
with open(train_file_path, 'r') as f:
    names_row = f.readline();
    csv_columns = names_row.rstrip('\n').split(',')
    print(csv_columns)

def get_dataset(file_path):
#     dataset_s = tf.data.experimental.CsvDataset(
#         filenames=file_path,
#         record_defaults=[tf.string, tf.float32, tf.string, tf.float32, tf.float32, tf.int32],
#         header=True,
#         field_delim=",",
#         select_cols=[0,1,2,3,4,5]
#     )
#     columns = ['name','cpn','rate','price','yield','result']
#     drops = ['price']
#     columns_use = [col for col in columns if col not in drops]
#     columns = ['Name','cpn','rate']
#     csv_columns = [col for col in columns]
    dataset_s = tf.data.experimental.make_csv_dataset(
        file_pattern=file_path,
        batch_size=12,
#         column_names=columns_use,
        label_name='result',
        num_epochs=1
    )
    return dataset_s

def process_categorical_data(data, categories):
    data = tf.strings.regex_replace(data, '^ ', '')
    data = tf.strings.regex_replace(data, r'\.$', '')
    data = tf.reshape(data, [-1, 1])
    data = tf.equal(categories, data)
    data = tf.cast(data, tf.float32)
    return data

def process_continuous_data(data, mean):
    data = tf.cast(data, tf.float32) * 1/(2 * mean)
    return tf.reshape(data, [-1, 1])

raw_train_data = get_dataset(train_file_path)
# print(raw_train_data)
# name, cpn, rate, price, yld, result = next(iter(raw_train_data))
# print("name:\n", name, "\n")
# print("cpn:\n", cpn, "\n")
# print("rate:\n", rate, "\n")
# print("price:\n", price, "\n")
# print("yld:\n", yld, "\n")
# print("result:\n", result)
examples, labels = next(iter(raw_train_data))
# print("examples:\n", examples, "\n")
# print("labels:\n", labels)

categories = {
    'rate': ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C'],
    'Name': ['Michael', 'Wendy', 'Vicky', 'Sam', 'George', 'Rose']
}
# rate_tenor = examples['rate']
# rate_category = categories['rate']
# processed_rate = process_categorical_data(rate_tenor, rate_category)
# print(processed_rate)
# 
# cpn_tenor = examples['cpn']
# cpn_mean = tf.keras.backend.mean(x=cpn_tenor)
# print(cpn_mean)
# processed_cpn = process_continuous_data(cpn_tenor, cpn_mean)
# print(processed_cpn)
def preprocess(features, labels):
    columns = ['Name','cpn','rate','price','yield']
    features['Name'] = process_categorical_data(examples['Name'], categories['Name'])
    features['cpn'] = process_continuous_data(examples['cpn'], tf.keras.backend.mean(x=examples['cpn']))
    features['rate'] = process_categorical_data(examples['rate'], categories['rate'])
    features['price'] = process_continuous_data(examples['price'], tf.keras.backend.mean(x=examples['price']))
    features['yield'] = process_continuous_data(examples['yield'], tf.keras.backend.mean(x=examples['yield']))
    features = tf.concat([features[column] for column in columns], -1)
    return features, labels

features={}
train_data = raw_train_data.map(preprocess).shuffle(500)
examples, labels = next(iter(train_data))
print("examples:\n", examples, "\n")
print("labels:\n", labels)
