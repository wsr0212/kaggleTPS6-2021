import pandas as pd
import numpy as np

#用了很笨的方法 载入数据
data = pd.read_csv('train.csv')
#print(data['target'].replace('Class_8',8))
data['target'].replace('Class_9',int(9),inplace=True)
data['target'].replace('Class_8',int(8),inplace=True)
data['target'].replace('Class_7',int(7),inplace=True)
data['target'].replace('Class_6',int(6),inplace=True)
data['target'].replace('Class_5',int(5),inplace=True)
data['target'].replace('Class_4',int(4),inplace=True)
data['target'].replace('Class_3',int(3),inplace=True)
data['target'].replace('Class_2',int(2),inplace=True)
data['target'].replace('Class_1',int(1),inplace=True)
#data['target'].to_numeric()
data['target'].astype(int)
data_0 = pd.read_csv('test.csv')

train_model = 

