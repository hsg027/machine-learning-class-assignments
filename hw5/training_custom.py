#Python 3.8.1 (tags/v3.8.1:1b293b6, Dec 18 2019, 22:39:24) [MSC v.1916 32 bit (Intel)] on win32
#Type "help", "copyright", "credits" or "license()" for more information.
# coding: utf-8

# In[2]:


import pickle
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.neighbors import KNeighborsClassifier



# In[3]:


def create_knn(fold_num,max_neighbors):
    x_train,y_train = read_train_csv(fold_num)
    for num in range(0,max_neighbors):
        name_1= "knn/knn_model_"+str(fold_num)+"k="+str(num)+".txt"
        knn = KNeighborsClassifier(n_neighbors = num+1)
        knn.fit(x_train, y_train)
        with open(name_1, 'wb') as f1:
            pickle.dump(knn, f1)
        f1.close()


    # knn = KNeighborsClassifier(n_neighbors=n)
    # knn.fit(x_train,y_train)
    # name_1= "knn/knn_model_"+str(fold_num)+"n="+str()+".txt"
    # with open(name_1, 'wb') as f1:
    #     pickle.dump(knn, f1)
    # f1.close()


# In[1]:


def create_dtc(fold_num):
    x_train,y_train = read_train_csv(fold_num)
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)
    name_2 = "dtc/dtc_model_"+str(fold_num)+".txt"
    with open(name_2, 'wb') as f2:
        pickle.dump(dtc, f2)
    f2.close()


# In[ ]:
def read_train_csv(num):
    x_train = np.genfromtxt("C:/Users/Harsh/Desktop/hw5/train.csv/x_train_"+str(num)+".csv",delimiter=",")
    y_train = np.genfromtxt("C:/Users/Harsh/Desktop/hw5/train.csv/y_train_"+str(num)+".csv",delimiter=",")
    return x_train,y_train





# In[ ]:





# In[ ]:



