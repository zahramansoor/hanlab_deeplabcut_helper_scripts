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
normnose=normalize([df['nose_y'].astype('float32').values])[0]


plt.scatter(range(39998),normblinks)
plt.scatter(range(39998),normnose)
plt.scatter(range(39998),normtongue)

from sklearn.cluster import KMeans

X = np.array([normblinks[:20000], normnose[:20000], normtongue[:20000]]).T
km = KMeans(
    n_clusters=3, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
).fit(X)


plt.scatter(
    X[y_km == 0, 0], X[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    X[y_km == 1, 0], X[y_km == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    X[y_km == 2, 0], X[y_km == 2, 1],
    s=50, c='lightblue',
    marker='v', edgecolor='black',
    label='cluster 3'
)

# plot the centroids
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.show()

