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

