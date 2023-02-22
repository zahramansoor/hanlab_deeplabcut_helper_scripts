# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 14:13:27 2023

@author: Han
"""

import pandas as pd, matplotlib.pyplot as plt, numpy as np

df = pd.read_csv(r'C:\Users\Han\Desktop\Licks_nose_blink-Zahra-2023-02-16\videos\200929_E130DLC_resnet50_Licks_nose_blinkFeb16shuffle1_10000.csv')
#change column names
df.columns=['bodyparts', 'eyeLidTop_x', 'eyeLidTop_y', 'eyeLidTop_likelihood', 'eyeLidBottom_x',
       'eyeLidBottom_y', 'eyeLidBottom_likelihood', 'nose_x', 'nose_y', 'nose_likelihood', 'tongue1_x',
       'tongue1_y', 'tongue1_likelihood', 'tongue2_x', 'tongue2_y', 'tongue2_likelihood', 'tongue3_x',
       'tongue3_y', 'tongue3_likelihood']#np.squeeze(np.array(df.iloc[[0]]))
df=df.drop([0,1])

#plot blinks
#here i think y pos starts from above
plt.plot(df['eyeLidBottom_y'].astype('float32').values - df['eyeLidTop_y'].astype('float32').values)

#plot nose movement
plt.plot(df['nose_y'].astype('float32').values)

#plot tongue1 movement
#assign to nans/0
keep1=df['tongue1_likelihood'].astype('float32') > 0.9
df['tongue1_x'][~keep1]=0
keep2=df['tongue2_likelihood'].astype('float32') > 0.9
df['tongue2_x'][~keep2]=0
keep3=df['tongue3_likelihood'].astype('float32') > 0.9
df['tongue3_x'][~keep3]=0


plt.plot(df['tongue1_x'].astype('float32').values)
plt.plot(df['tongue2_x'].astype('float32').values)
plt.plot(df['tongue3_x'].astype('float32').values)

#k means
from sklearn.preprocessing import normalize
#data
blinks=df['eyeLidBottom_y'].astype('float32').values - df['eyeLidTop_y'].astype('float32').values
normblinks=normalize([blinks])[0]
#tongue movement
tongue=df[['tongue1_x','tongue2_x','tongue3_x']].astype('float32').mean(axis=1, skipna=False)
normtongue=normalize([tongue])[0]
#nose
nose=df['nose_y'].astype('float32').values
normnose=normalize([nose])[0]

plt.scatter(range(39998),normblinks)
plt.scatter(range(39998),normnose)
plt.scatter(range(39998),normtongue)

#%%
# PCA and kmeans
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1
df = pd.DataFrame(np.array([blinks,nose,tongue]).T)
df.columns=['blinks','nose','tongue_av']
X_scaled=StandardScaler().fit_transform(df)
#https://medium.com/swlh/k-means-clustering-on-high-dimensional-data-d2151e1a4240
pca_2 = PCA(n_components=2)
pca_2_result = pca_2.fit_transform(X_scaled)
print('Explained variation per principal component: {}'.format(pca_2.explained_variance_ratio_))
print('Cumulative variance explained by 2 principal components: {:.2%}'.format(np.sum(pca_2.explained_variance_ratio_)))
#convert to df...
X_scaled = pd.DataFrame(X_scaled, columns=df.columns)

dataset_pca = pd.DataFrame(abs(pca_2.components_), columns=X_scaled.columns, index=['PC_1', 'PC_2'])
print('\n\n', dataset_pca)

print("\n*************** Most important features *************************")
print('As per PC 1:\n', (dataset_pca[dataset_pca > 0.3].iloc[0]).dropna())   
print('\n\nAs per PC 2:\n', (dataset_pca[dataset_pca > 0.3].iloc[1]).dropna())
print("\n******************************************************************")

# 4. Hyperparameter tuning using the silhouette score method
# Apart from the curse of dimensionality issue, KMeans also has this problem where we need to explicitly inform the KMeans model about the number of clusters we want our data to be categorised in, this hit and trial can be daunting, so we are using silhouette score method. Here you give a list of probable candidates and the metrics.silhouette_score method calculates a score by applying the KMeans model to our data considering one value (number of clusters) at a time. For eg., if we want to check how good our model will be if we ask it to form 2 clusters out of our data, we can check the silhouette score for clusters=2.

# Silhouette score value ranges from 0 to 1, 0 being the worst and 1 being the best.

# candidate values for our number of cluster
parameters = [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40]
# instantiating ParameterGrid, pass number of clusters as input
parameter_grid = sk.model_selection.ParameterGrid({'n_clusters': parameters})
best_score = -1
kmeans_model = KMeans()     # instantiating KMeans model
silhouette_scores = []
# evaluation based on silhouette_score
for p in parameter_grid:
    kmeans_model.set_params(**p)    # set current hyper parameter
    kmeans_model.fit(df)          # fit model on wine dataset, this will find clusters based on parameter p
    ss = sk.metrics.silhouette_score(df, kmeans_model.labels_)   # calculate silhouette_score
    silhouette_scores += [ss]       # store all the scores
    print('Parameter:', p, 'Score', ss)
    # check p which has the best score
    if ss > best_score:
        best_score = ss
        best_grid = p
# plotting silhouette score
plt.bar(range(len(silhouette_scores)), list(silhouette_scores), align='center', color='#722f59', width=0.5)
plt.xticks(range(len(silhouette_scores)), list(parameters))
plt.title('Silhouette Score', fontweight='bold')
plt.xlabel('Number of Clusters')
plt.show()

# fitting KMeans    
kmeans = KMeans(n_clusters=3)    
kmeans.fit(pca_2_result)
label = kmeans.fit_predict(pca_2_result)

# plot pc components
uniq = np.unique(label)
for i in uniq:
   plt.scatter(pca_2_result[label == i , 0] , pca_2_result[label == i , 1] , label = i)

#plt.scatter(pca_2_result[:,0],pca_2_result[:,1],s=10,color='k')
#plot kmeans centroids (first 2 dim??? or only after run on pca)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s=50,color='k',marker='x')