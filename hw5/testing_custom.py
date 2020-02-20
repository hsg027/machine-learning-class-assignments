#Python 3.8.1 (tags/v3.8.1:1b293b6, Dec 18 2019, 22:39:24) [MSC v.1916 32 bit (Intel)] on win32
#Type "help", "copyright", "credits" or "license()" for more information.
# coding: utf-8

# In[9]:


import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np


# In[7]:


def run_knn(fold_num,max_neighbors):
    x_test,y_test = read_test_csv(fold_num)
    accuracies = np.zeros(max_neighbors)
    accuracies = accuracies.astype(np.float)
    for num in range(0,max_neighbors):
        name_1 = "knn/knn_model_"+str(fold_num)+"k="+str(num)+".txt"
        with open(name_1, 'rb') as f3:
            loaded_model = pickle.load(f3)
        f3.close()
        y_pred = loaded_model.predict(x_test)
        accuracy =  accuracy_score(y_test,y_pred)
        accuracies[num] = accuracy.astype(np.float64) 
        file_object = open('accuracy/accuracy_knn.txt', 'a')
        file_object.write("Accuracy For knn for fold " +str(fold_num) +" and k = "+ str(num+1) +" is : "+ str(accuracy) + "\n")
        file_object.close()
    return accuracies


# In[8]:


def run_dtc(fold_num):
    x_test,y_test = read_test_csv(fold_num)
    name_2 = "dtc/dtc_model_"+str(fold_num)+".txt"
    with open(name_2, 'rb') as f4:
        loaded_model = pickle.load(f4) 
    f4.close()
    y_pred = loaded_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    file_object = open('accuracy/accuracy_dtc.txt', 'a')
    file_object.write("Accuracy For dtc for fold " +str(fold_num) +" is : "+ str(accuracy) + "\n")
    file_object.close()
    return accuracy 


# In[ ]:
def read_test_csv(num):
    x_test = np.genfromtxt("C:/Users/Harsh/Desktop/hw5/test.csv/x_test_"+str(num)+".csv",delimiter=",")
    y_test = np.genfromtxt("C:/Users/Harsh/Desktop/hw5/test.csv/y_test_"+str(num)+".csv",delimiter=",")
    return x_test,y_test




