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
#数据预处理-->MinMax
#from sklearn.preprocessing import MinMaxScaler
#print(StandardScaler().fit_transform(data.loc[:,'feature_0':'feature_74']))
#data.loc[:,'feature_0':'feature_74'] = MinMaxScaler().fit_transform(data.loc[:,'feature_0':'feature_74'])

#数据预处理-->标准化
#from sklearn.preprocessing import StandardScaler
#print(StandardScaler().fit_transform(data.loc[:,'feature_0':'feature_74']))
#data.loc[:,'feature_0':'feature_74'] = StandardScaler().fit_transform(data.loc[:,'feature_0':'feature_74'])

#效果都不好,最终没用

#降维-->PCA
#from sklearn.decomposition import PCA
#pca = PCA(n_components=27)

#降维-->LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=7)
X_train = data.loc[:,'id':'feature_74']
y_train = data['target']
lda.fit(X_train,y_train)
X_train = lda.transform(X_train)
X_test = data_0.loc[:,'id':'feature_74']
#lda(n_components=7).fit_transform(X_train,y_train)
#print(X_train)
#print(X_train.shape)

