# Importing necessary packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import scipy.spatial.distance as dista
from sklearn import metrics
import random
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from scipy import stats
from copy import deepcopy
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

PATH="C:/Users/harsh/Desktop/Sem-2/ML/CS5402-master/homework5/"
data = pd.read_csv(PATH + 'football.csv')
print(data.shape)
data.head()

f1 = data['winsIn2016'].to_numpy()
f2 = data['winsIn2017'].to_numpy()
X = np.column_stack((f1, f2))
plt.scatter(f1, f2)

k=2
C_1 = np.array([4,5])
C_2 = np.array([6,4])
C = np.column_stack([C_1, C_2])
colors = ['r', 'g', 'b', 'y', 'c', 'm']

#definitions

def dist(a, b, ax=1, metric='e'):
    switcher={
        'm':np.sum(np.abs(a-b), axis=ax),
        'e':np.sum((a-b)**2, axis=ax),
        'c':cosine_sim(a,b,metric),
        'j':(1-np.sum(np.minimum(a,b),axis=ax)/np.sum(np.maximum(a,b),axis=ax))
    }
    return switcher.get(metric)

#k-means definition for the case when SSE value increases in second iteration
def kmeans(X, Centroid=C, k=2, kmeans_metric='m'):
    
    max_iter = 100
    np.random.seed(89)
    
    if Centroid is None:
        Centroid = X[np.random.choice(len(X), size=k, replace=False)]
    
    # Temprarily store Centroid values
    old_C = np.ones(Centroid.shape)
    
    # Cluster Lables
    clusters = np.zeros(len(X))
    
    # Error func. - Distance between new centroids and old centroids  
    err = np.array(distance(Centroid, old_C, None, metric=kmeans_metric))

    print(err)
    count = 1
    sse_prev = 0.5
    sse_curr = 0
    
    while (err.any() != 0 and sse_prev>sse_curr and count<=max_iter):
        
        # Assigning each value to its closest cluster
        for i in range(len(X)):
            dist = distance(X[i], Centroid,1,kmeans_metric)
            clusters[i] = np.argmin([dist])
                         
        # Storing the old centroid values
        old_C = deepcopy(Centroid)
        sse_curr = sse(X, clusters, Centroid)
        print('Iteration: ' + str(count) + ' Current SSE: ' + str(sse_curr) + ' Previous SSE: ' + str(sse_prev))
        
        # Finding the new centroids by taking the average value
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            Centroid[i] = np.mean(points, axis=0) 
        
        err_old = deepcopy(err)
        err = distance(Centroid, old_C, None,kmeans_metric)
        
        if count>0:
            if np.sum(err_old) == np.sum(err):
                break
        count= count+1
        sse_prev = sse_curr
    return clusters, count

#k-means definition for default cases when there is no change in centroid position and when max preset value (100) is complete
def k_means(X, Centroid=C, k=2, kmeans_metric='m'):
    
    max_iter = 100
    np.random.seed(89)
    
    if Centroid is None:
        Centroid = X[np.random.choice(len(X), size=k, replace=False)]
    
    # Temprarily store Centroid values
    old_C = np.ones(Centroid.shape)
    
    # Cluster Lables
    clusters = np.zeros(len(X))
    
    # Error func. - Dist between new centroids and old centroids  
    err = np.array(dist(Centroid, old_C, None, metric=kmeans_metric))

    print(err)
    count = 1
    sse_prev = 0
    sse_curr = 0
    
    
    while (err.any() != 0 and count<=max_iter):
        
        # Assigning each value to its closest cluster
        for i in range(len(X)):
            dist = dist(X[i], Centroid,1,kmeans_metric)
            clusters[i] = np.argmin([dist])
                         
        # Storing the old centroid values
        old_C = deepcopy(Centroid)
        sse_curr = sse(X, clusters, Centroid)
        print('Iteration: ' + str(count) + ' Current SSE: ' + str(sse_curr) + ' Previous SSE: ' + str(sse_prev))
        
        # Finding the new centroids by taking the average value
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            Centroid[i] = np.mean(points, axis=0) 
        
        err_old = deepcopy(err)
        err = dist(Centroid, old_C, None,kmeans_metric)
        
        if count>0:
            if np.sum(err_old) == np.sum(err):
                break
        count= count+1
        sse_prev = sse_curr
    return clusters, count


def football(C_x, C_y,metric):
    fig, ax = plt.subplots()
    
    C = np.column_stack((C_x,C_y))
    # Plotting along with the Centroids
    plt.scatter(f1, f2, c='#050505')
    plt.scatter(C_x, C_y, marker='*', s=200, c='y')
    clust, count = kmeans(X, Centroid=C, k=2,kmeans_metric=metric)
    print('Number of count: '+str(count))
    for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clust[j] == i])
        ax.scatter(points[:, 0], points[:, 1], c=colors[i])
        
    ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
    ax.legend(["","old centroids","clust 1","clust 2","new centroids"])
    


def sse(X, clusters, C, metric='e'):
    err = 0
    for i, centroid in enumerate(C):
        err += np.sum(dist(X[np.where(clusters==i)], centroid,ax=1,metric='e'))
    
    return err

def predict(clusters, y, k=3):
    indexes = []
    for i in range(k):
        indexes.append(np.where(clusters == i))
    for cluster in indexes:
        mode = int(stats.mode(y[cluster])[0])
        clusters[cluster] = mode
        
    return clusters

def iris():
    fig, ax = plt.subplots()
    for i in range(3):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], c=colors[i])
        
def print_accuracy():
    pred_val = predict(clusters, df['class'].values)
    accurcy = metrics.accuracy_score(df['class'].values, pred_val)
    print ("The original clusters  are ")
    print(df['class'].values)
    print ("The predicted clusters are ")
    print(pred_val)
    print("accuracy is " + np.array2string(accurcy, formatter={'float_kind':lambda x: "%.5f" % x}))

    
def cosine_sim(a,b,m):
    if m=='c':
        c=0
        if a.ndim != 1:
            for i in range(3): 
                c=c+dista.cosine(a[i],b[i])
            return c
        else :
            ci=[0,0,0]
            for i in range(3):
                ci[i]=dista.cosine(a,b[i])
            return np.asarray(ci, dtype=np.float32)
    return 0
 
# Q2, part 1 
# Number of clusters
k = 2

# X coordinates of random centroids
C_x = np.array([4,5])
# Y coordinates of random centroids
C_y = np.array([6,4])

football(C_x, C_y,metric='e')

#Q2, part 3
# Number of clusters
k = 2

# X coordinates of random centroids
C_x = np.array([3,8])
# Y coordinates of random centroids
C_y = np.array([3,3])

football(C_x, C_y,metric='m')

#Q2, part 4
# Number of clusters
k = 2

# X coordinates of random centroids
C_x = np.array([3,4])
# Y coordinates of random centroids
C_y = np.array([2,8])

football(C_x, C_y,metric='m')

#Q2, part 2
# Number of clusters
k = 2

# X coordinates of random centroids
C_x = np.array([4,5])
# Y coordinates of random centroids
C_y = np.array([6,4])

football(C_x, C_y,metric='m')

#Q3 IRIS Data sets

df = pd.read_table(PATH+ "iris.data", sep=",", header=None, names=['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth', 'class'])
# Converting the predicted label "class" to numerical values
df['class'] = pd.Categorical(df['class'])
df['class'] = df['class'].cat.codes
df.head()

X = df[df.columns[:-1]].values
# X[1].shape
plt.scatter(X[:, 0], X[:, 1], c='black')

clusters, count = kmeans(X, Centroid=None, k=3,kmeans_metric='e')
print("number of count is ", str(int(count)))
iris()
print_accuracy()

clusters,count = kmeans(X, Centroid=None, k=3,kmeans_metric='j')
print("number of count is ", str(int(count)))
iris()
print_accuracy()

clusters,count = kmeans(X, Centroid=None, k=3,kmeans_metric='c')
print("number of count is ", str(int(count)))
iris()
print_accuracy()
