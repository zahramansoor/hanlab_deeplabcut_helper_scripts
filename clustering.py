# https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
# Unfortunately, the k-means model has no intrinsic measure of probability or uncertainty of cluster 
# assignments (although it may be possible to use a bootstrap approach to estimate this uncertainty). 
# For this, we must think about generalizing the model.
# An important observation for k-means is that these cluster models must be circular: 
# k-means has no built-in way of accounting for oblong or elliptical clusters.
import pandas as pd, matplotlib.pyplot as plt, numpy as np, seaborn as sns
import scipy.ndimage
from scipy.io import loadmat
from sklearn.mixture import GaussianMixture
#%%
df = pd.read_csv(r'Y:\DLC\DLC_networks\pupil_licks_nose_paw-Zahra-2023-02-27\videos\230301_E201DLC_resnet50_pupil_licks_nose_pawFeb27shuffle1_50000.csv')
mat = loadmat(r'Z:\sstcre_imaging\e201\10\230301_ZD_000_001\suite2p\plane0\Fall.mat') # load fall with behavior aligned data
forwardvelocity = mat['forwardvel'][0]

#change column names
cols=[[xx+"_x",xx+"_y",xx+"_likelihood"] for xx in pd.unique(df.iloc[0]) if xx!="bodyparts"]
cols = [yy for xx in cols for yy in xx]; cols.insert(0, 'bodyparts')
df.columns = cols
df=df.drop([0,1])

#plot tongue1 movement
#assign to nans/0
keep1=df['tongue1_likelihood'].astype('float32') > 0.9
df['tongue1_x'][~keep1]=0
keep2=df['tongue2_likelihood'].astype('float32') > 0.9
df['tongue2_x'][~keep2]=0
keep3=df['tongue3_likelihood'].astype('float32') > 0.9
df['tongue3_x'][~keep3]=0
#do for lip 1 since it is jittery in inference
keep4=df['lip1_likelihood'].astype('float32') > 0.9
df['lip1_x'][~keep4]=0


blinks=scipy.ndimage.gaussian_filter(df['eyeBottom_y'].astype('float32').values - df['eyeTop_y'].astype('float32').values,sigma=3)
#tongue movement
tongue = df['tongue1_x'].astype('float32').values

#nose
nose=df[['noseTop_y','noseBottom_y']].astype('float32').mean(axis=1, skipna=False).astype('float32').values
#lip movement/mouth open
mouth_open=df['lip2_x'].astype('float32').values
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
#https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1
dfkmeans = pd.DataFrame(np.array([blinks[1::2], #binning by taking every other value, is this the right way???
    nose[1::2],tongue[1::2],mouth_open[1::2], forwardvelocity]).T)

columns = ['blinks','nose','tongue','mouth_open', 'velocity']
dfkmeans.columns=columns
plt.figure()
plt.plot(dfkmeans['blinks'])
plt.axhline(y=dfkmeans["blinks"].mean()+dfkmeans["blinks"].std()*4, color='r', linestyle='-')
plt.figure()
plt.plot(dfkmeans['nose'])
plt.axhline(y=dfkmeans["nose"].mean()+dfkmeans["nose"].std()*4, color='r', linestyle='-')
plt.figure()
plt.plot(dfkmeans['mouth_open'])
plt.axhline(y=dfkmeans["mouth_open"].mean()+dfkmeans["mouth_open"].std()*4, color='r', linestyle='-')
plt.figure()
plt.plot(dfkmeans['velocity'])
plt.axhline(y=dfkmeans["velocity"].mean()+dfkmeans["velocity"].std()*4, color='r', linestyle='-')

#classify blinks, sniffs, licks?
dfkmeans['blinks_lbl'] = dfkmeans['blinks']>dfkmeans["blinks"].mean()+dfkmeans["blinks"].std()*5 
dfkmeans['sniff_lbl'] =  dfkmeans['nose']>dfkmeans["nose"].mean()+dfkmeans["nose"].std()*4
dfkmeans['licks'] =  dfkmeans['tongue']>0#arbitrary thres
#dfkmeans['mouth_open1'] =  [True if xx > 298 else False for i,xx in enumerate(dfkmeans['mouth_open1'])] #arbitrary thres
dfkmeans['mouth_mov'] =  dfkmeans['mouth_open']>dfkmeans["mouth_open"].mean()+dfkmeans["mouth_open"].std()*4 #arbitrary thres
dfkmeans['fastruns'] =  dfkmeans['velocity']>dfkmeans["velocity"].mean()+dfkmeans["velocity"].std()*4 #arbitrary thres
#stopped for 5 seconds around the cell
dfkmeans['stops'] =  [True if sum(dfkmeans["velocity"].iloc[xx-5:xx+5])==0 else False for xx in range(len(dfkmeans["velocity"]))]

X_scaled=StandardScaler().fit_transform(dfkmeans[columns])#,'mouth_open1','mouth_open2']])
#https://medium.com/swlh/k-means-clustering-on-high-dimensional-data-d2151e1a4240
pca_2 = PCA(n_components=2)
pca_2_result = pca_2.fit_transform(X_scaled)
print('Explained variation per principal component: {}'.format(pca_2.explained_variance_ratio_))
print('Cumulative variance explained by 2 principal components: {:.2%}'.format(np.sum(pca_2.explained_variance_ratio_)))
#convert to df...
X_scaled = pd.DataFrame(X_scaled, columns=columns)#,'mouth_open1','mouth_open2'])

dataset_pca = pd.DataFrame(abs(pca_2.components_), columns=X_scaled.columns, index=['PC_1', 'PC_2'])
print('\n\n', dataset_pca)

print("\n*************** Most important features *************************")
print('As per PC 1:\n', (dataset_pca[dataset_pca > 0.3].iloc[0]).dropna())   
print('\n\nAs per PC 2:\n', (dataset_pca[dataset_pca > 0.3].iloc[1]).dropna())
print("\n******************************************************************")

#%%
#kmeans
# candidate values for our number of cluster
parameters = np.linspace(2,100,99).astype(int)
# instantiating ParameterGrid, pass number of clusters as input
parameter_grid = sk.model_selection.ParameterGrid({'n_clusters': parameters})
best_score = -1
kmeans_model = KMeans()     # instantiating KMeans model
silhouette_scores = []
# evaluation based on silhouette_score
for p in parameter_grid:
    kmeans_model.set_params(**p)    # set current hyper parameter
    kmeans_model.fit(X_scaled)          # fit model on wine dataset, this will find clusters based on parameter p
    ss = sk.metrics.silhouette_score(X_scaled, kmeans_model.labels_)   # calculate silhouette_score
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
kmeans.fit(X_scaled)
label = kmeans.fit_predict(X_scaled)

# plot pc components
plt.scatter(pca_2_result[:, 0],
                pca_2_result[: , 1], c=label, cmap='viridis', alpha=0.3)
#plot behaviors
pca_2_result_bl=pca_2_result[dfkmeans['blinks_lbl']]
plt.scatter(pca_2_result_bl[:, 0] , pca_2_result_bl[: , 1] , color='k', marker='+')
pca_2_result_sn=pca_2_result[dfkmeans['sniff_lbl']]
plt.scatter(pca_2_result_sn[:, 0] , pca_2_result_sn[: , 1] , 
            color='k', marker='x')
pca_2_result_lk=pca_2_result[dfkmeans['licks']]
plt.scatter(pca_2_result_lk[:, 0] , pca_2_result_lk[: , 1] , 
            color='k', marker='o', facecolors='none')
# pca_2_result_mo=pca_2_result[dfkmeans['mouth_mov']]
# plt.scatter(pca_2_result_mo[:, 0] , pca_2_result_mo[: , 1] , 
#             color='k', marker='d', facecolors='none')
pca_2_result_fast=pca_2_result[dfkmeans['fastruns']]
plt.scatter(pca_2_result_fast[:, 0] , pca_2_result_fast[: , 1] , 
            color='k', marker='s', facecolors='none')
pca_2_result_stop=pca_2_result[dfkmeans['stops']]
plt.scatter(pca_2_result_stop[:, 0] , pca_2_result_stop[: , 1] , 
            color='k', marker='|')

plt.legend(['Clusters', 'blink', 
            'sniff', 'lick', 'runs', 'stops'])
plt.xlabel("PC1")
plt.ylabel("PC2")


#%%
gmm = GaussianMixture(n_components=3).fit(X_scaled)
label = gmm.predict(X_scaled)
probs = gmm.predict_proba(X_scaled)

size = 50 * probs.max(1) ** 2  # square emphasizes differences

# plot pc components
plt.scatter(pca_2_result[:, 0],
                pca_2_result[: , 1], c=label, cmap='viridis', s=size, alpha=0.3)
#plot behaviors
pca_2_result_bl=pca_2_result[dfkmeans['blinks_lbl']]
plt.scatter(pca_2_result_bl[:, 0] , pca_2_result_bl[: , 1] , color='k', marker='+')
pca_2_result_sn=pca_2_result[dfkmeans['sniff_lbl']]
plt.scatter(pca_2_result_sn[:, 0] , pca_2_result_sn[: , 1] , 
            color='k', marker='x')
pca_2_result_lk=pca_2_result[dfkmeans['licks']]
plt.scatter(pca_2_result_lk[:, 0] , pca_2_result_lk[: , 1] , 
            color='k', marker='o', facecolors='none')
# pca_2_result_mo=pca_2_result[dfkmeans['mouth_mov']]
# plt.scatter(pca_2_result_mo[:, 0] , pca_2_result_mo[: , 1] , 
#             color='k', marker='d', facecolors='none')
pca_2_result_fast=pca_2_result[dfkmeans['fastruns']]
plt.scatter(pca_2_result_fast[:, 0] , pca_2_result_fast[: , 1] , 
            color='k', marker='s', facecolors='none')
pca_2_result_stop=pca_2_result[dfkmeans['stops']]
plt.scatter(pca_2_result_stop[:, 0] , pca_2_result_stop[: , 1] , 
            color='k', marker='|')

plt.legend(['Clusters', 'blink', 
            'sniff', 'lick', 'runs', 'stops'])
plt.xlabel("PC1")
plt.ylabel("PC2")
