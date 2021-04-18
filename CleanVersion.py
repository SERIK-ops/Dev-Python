

# Author Mohamed SERIK

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pymf3 as P
from sklearn import metrics
from sklearn.metrics.cluster import *
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel,pairwise_distances, polynomial_kernel


"****** PARTIE 1 ******"
"loading of the data set"
data = np.loadtxt("bank_auth.txt",delimiter=",")


"transofrming the data into a matrix"
data = np.array(data)


"****** QUESTION A-1 ******"

"saving the data in bank_data"
bank_data = data[:,:-1]
print("****** bank_data ******")
print(bank_data)

"saving the labels in bank_labels"
bank_labels = data[:,4 ]
print("****** bank_labels ******")
print(bank_labels)


# In[74]:


"****** QUESTION 1-B ******"

"matrix with only data of fake money bills"
A_faux  = bank_data[bank_labels == 0]
print("****** A_faux ******")
print(A_faux)

"matrix with only data of real money bills"
A_vrais = bank_data[bank_labels == 1 ]
print("****** A_vrais  ******")
print(A_vrais)


# In[111]:


"****** QUESTION 1-C ******"

"matrix with 500 data of real money bills and  600 data of fake money bills"
bank_train = np.concatenate((A_vrais[0:500,:],A_faux[0:600,:]))
print("****** bank_train ******")
print(bank_train)

"bank train labels"
bank_train_labels = np.zeros((1,1100))
bank_train_labels[:,0:500] = 1;
bank_train_labels = bank_train_labels.T
print("****** bank_train_labels ******")
print(bank_train_labels)

"the rest of the data"
bank_test = np.concatenate((A_vrais[500:,:],A_faux[600:,:]))
print("****** bank_test ******")
print(bank_test)

"bank test labels"
a = A_vrais.shape[0] - 500 
b = A_faux.shape[0] - 600
bank_test_labels = np.zeros((1,a+b))
bank_test_labels[:,0:a] = 1; 
bank_test_labels = bank_test_labels.T
print("****** bank_test_labels ******")
print(bank_test_labels)


# In[112]:


"****** QUESTION 1-D ******"
"****** Visualisation des données ******"

"Principale compenents"
pca = PCA(n_components=2)
bank_data1 = pca.fit_transform(bank_data)
bank_data1 = pca.transform(bank_data)


"scattering the data"
plt.scatter(bank_data1[:,0],bank_data1[:,1],s=10, c=bank_labels)

#plt.show()


# In[113]:


"****** PARTIE 2 ******"
"****** Application de la semi-nmf ******"
semi_nmf = P.semiNMF(bank_train.T, num_base = 2)

semi_nmf.factorize()

W_train1 = semi_nmf.W
H_train1 = semi_nmf.H
print("****** W_train1 ******")
print(W_train1)
print("****** H_train1 ******")
print(H_train1)


# In[114]:


"****** QUESTION 2-A ******"

def show_clusters(x):
    # get number of rows and columns 
    t=np.shape(x)
      
    for i in range(0,t[0]):
        p=np.max(x[i])
        for j in range(0,t[1]):
            if x[i][j]==p:
                x[i][j]=1
            else :
                x[i][j]=0
    return(x)


# In[115]:


show_clusters(H_train1)
I1 = H_train1[1,:]
print("****** I1 ******")
print(I1)

"****** La pureté de la matrice de partition  I1 ******"
accuracy_score_I = accuracy_score(I1,bank_train_labels)

print("la pureté de I1 : ", accuracy_score_I)


# In[116]:


"****** QUESTION 2-B ******"

H_test1 = np.dot(np.linalg.pinv(W_train1),bank_test.T)
print("****** H_test1 ******")
print(H_test1)

H_labels1 = show_clusters(H_test1.T)

H_labels1 = H_labels1[:,1]
print("****** H_labels1 ******")
print(H_labels1)


# In[127]:


"****** QUESTION 2-C ******"

"****** La pureté de H_labels1 ******"
accuracy_score_SEMI_NMF = accuracy_score(H_labels1,bank_test_labels)
print("La pureté de H_labels1:",accuracy_score_SEMI_NMF )

"****** L'entropie de H_labels1 ******"
entropy_SEMI_NMF = entropy(H_labels1)
print("L'entropie de H_labels1",entropy_SEMI_NMF)


# In[128]:


"****** QUESTION 2-D ******"
"****** Calcul des indices internes ******"

davies_bouldin_score_SEMI_NMF  = davies_bouldin_score(bank_test,H_labels1)
print(davies_bouldin_score_SEMI_NMF)
calinski_harabasz_score_SEMI_NMF = calinski_harabasz_score(bank_test,H_labels1)
print(calinski_harabasz_score_SEMI_NMF)


# In[129]:


"****** QUESTION 2-E ******"

"****** Visualisation des données ******"

"Principale compenents"
bank_test1 = pca.fit_transform(bank_test)
bank_test1 = pca.transform(bank_test)



plt.scatter(bank_test1[:,0],bank_test1[:,1],s=20, c= H_labels1)


# In[130]:


"****** PARTIE 3 ******"
"****** Application de la nmf ******"
def negative_to_positive(x):
    # get number of rows and columns 
    t=np.shape(x)
    m=np.min(x)  
    for i in range(0,t[0]):
        
        for j in range(0,t[1]):
            if m < 0:
                x[i][j]= x[i][j] - m
            else :
                x[i][j]= x[i][j] + m 
    return(x)


# In[131]:


nmf = P.NMF(negative_to_positive(bank_train).T,num_bases=2)

nmf.factorize()

W_train2 = nmf.W
H_train2 = nmf.H
print("****** W_train2 ******")
print(W_train2)
print("****** H_train2 ******")
print(H_train2)


# In[132]:


"****** QUESTION 3-A ******"

show_clusters(H_train2)
I2 = H_train2[1,:]
print("****** I2 ******")
print(I2)
accuracy_score_nmf = accuracy_score(I2,bank_train_labels)
print("la pureté de I2 : ",accuracy_score_nmf  )


# In[133]:


"****** QUESTION 3-B ******"

H_test2 = np.dot(np.linalg.pinv(W_train2),bank_test.T)
print("****** H_test2 ******")
print(H_test2)

H_labels2 =show_clusters(H_test2.T)

H_labels2 = H_labels2[:,0]
print("****** H_labels2 ******")
print(H_labels2)


# In[134]:


"****** QUESTION 3-C ******"

"****** La pureté de H_labels2 ******"
accuracy_score_NMF = accuracy_score(H_labels2,bank_test_labels)
print("La pureté de H_lables2:",accuracy_score_NMF )

"****** L'entropie de H_labels2 ******"
entropy_NMF = entropy(H_labels2)
print("L'entropie de H_labels2",entropy_NMF)


# In[135]:


"****** QUESTION 3-D ******"
davies_bouldin_score_NMF  = davies_bouldin_score(bank_test,H_labels2)
print(davies_bouldin_score_NMF)
calinski_harabasz_score_NMF = calinski_harabasz_score(bank_test,H_labels2)
print(calinski_harabasz_score_NMF)


# In[136]:


"****** QUESTION 3-E ******"

"****** Visualisation des données ******"
plt.scatter(bank_test1[:,0],bank_test1[:,1],s=20, c= H_labels2)


# In[139]:


"****** PARTIE 4 ******"
"****** Application de la Symmetric nmf ******"
"****** Implémentation de la Symmetric nmf d'abord ******"
def symNMF(A,r,niter=None):  
#W = W.*(0.5 + 0.5*(A*W)./(W*W'*W));       
    m=len(A)

# symNMF
    W=np.random.rand(m,r)

# symNMF
    for t in range(1,niter):
        # update At
        W=multiply(W,(0.5 + dot(0.5,(dot(A,W))) / (dot(dot(W,W.T),W))))

    return W


# In[140]:


"****** QUESTION 4-A1 ******"
"Application  avec paramétre 1"
k_test1 = rbf_kernel(bank_test,bank_test,1)

"****** QUESTION 4-B1 ******"
sym1 = symNMF(k_test1,2,1)
k1 = show_clusters(sym1)
k1 = k1[:,0]
accuracy_score_SYM_NMF1 =  accuracy_score(k1,bank_test_labels)
entropy_SYM_NMF1 = entropy(k1)

davies_bouldin_score_SYM_NMF1  = davies_bouldin_score(bank_test,k1)

calinski_harabasz_score_SYM_NMF1 = calinski_harabasz_score(bank_test,k1)


# In[141]:


"****** QUESTION 4-C1 ******"
"****** Visualisation des données ******"
plt.scatter(bank_test1[:,0],bank_test1[:,1],s=20, c= k1)


# In[142]:


"Application  avec paramétre 2"
"****** QUESTION 4-A2 ******"
k_test2 = rbf_kernel(bank_test,bank_test,2)
"****** QUESTION 4-B2 ******"
sym2 = symNMF(k_test2,2,1)
k2 = show_clusters(sym2)
k2 = k2[:,0]
accuracy_score_SYM_NMF2 = accuracy_score(k2,bank_test_labels)
entropy_SYM_NMF2 = entropy(k2)
davies_bouldin_score_SYM_NMF2  = davies_bouldin_score(bank_test,k2)

calinski_harabasz_score_SYM_NMF2 = calinski_harabasz_score(bank_test,k2)


# In[143]:


"****** QUESTION 4-C2 ******"
"****** Visualisation des données ******"
plt.scatter(bank_test1[:,0],bank_test1[:,1],s=20, c=k2)


# In[144]:


"Application  avec paramétre 3"
"****** QUESTION 4-A3 ******"
k_test3 = rbf_kernel(bank_test,bank_test,3)
"****** QUESTION 4-B3 ******"
sym3 = symNMF(k_test3,2,1)
k3 = show_clusters(sym3)
k3 = k3[:,1]
accuracy_score_SYM_NMF3 =  accuracy_score(k3,bank_test_labels)
entropy_SYM_NMF3 = entropy(k3)
davies_bouldin_score_SYM_NMF3 = davies_bouldin_score(bank_test,k3)

calinski_harabasz_score_SYM_NMF3 = calinski_harabasz_score(bank_test,k3)


# In[145]:


"****** QUESTION 4-C3 ******"
"****** Visualisation des données ******"
plt.scatter(bank_test1[:,0],bank_test1[:,1],s=20, c=k3)


# In[146]:


"Application  avec paramétre 1,0,2"

"****** QUESTION 4-A4 ******"

k_test4 = polynomial_kernel(bank_test,bank_test,1,0,2)
"****** QUESTION 4-B4 ******"

sym4 = symNMF(k_test4,2,1)
k4 = show_clusters(sym4)
k4 = k4[:,0]
accuracy_score_SYM_NMF4  = accuracy_score(k4,bank_test_labels)
entropy_SYM_NMF4 = entropy(k4)
davies_bouldin_score_SYM_NMF4  = davies_bouldin_score(bank_test,k4)

calinski_harabasz_score_SYM_NMF4 = calinski_harabasz_score(bank_test,k4)


# In[147]:


"****** QUESTION 4-C4 ******"
plt.scatter(bank_test1[:,0],bank_test1[:,1],s=20, c= k4)


# In[148]:


"****** PARTIE 5 ******"
"****** Comparaison des résultats ******"
SEMI_NMF = pd.Series({'purity':accuracy_score_SEMI_NMF,
                      'entropy':entropy_SEMI_NMF})
NMF = pd.Series({'purity':accuracy_score_NMF,
                      'entropy':entropy_NMF})
SYM_NMF1 = pd.Series({'purity':accuracy_score_SYM_NMF1,
                      'entropy':entropy_SYM_NMF1})
SYM_NMF2 = pd.Series({'purity':accuracy_score_SYM_NMF1,
                      'entropy':entropy_SYM_NMF2})
SYM_NMF3 = pd.Series({'purity':accuracy_score_SYM_NMF3,
                      'entropy':entropy_SYM_NMF3})
SYM_NMF4 = pd.Series({'purity':accuracy_score_SYM_NMF4,
                      'entropy':entropy_SYM_NMF4})


# In[108]:


indices = pd.DataFrame([SEMI_NMF, NMF, SYM_NMF1, SYM_NMF2, SYM_NMF3, SYM_NMF4], index = ['SEMI_NMF', 'NMF', 'SYM_NMF1', 'SYM_NMF2', 'SYM_NMF3', 'SYM_NMF4'])


# In[149]:


def highlight_max(data,color='yellow'):
    #highlight the maximum
    attr='background-color: {}'.format(color)
    data = data.replace('%','',regex=True).astype(float)
    if data.ndim == 1:
        is_max = data == data.max()
        return [attr if v else ''for v in is_max]
    else:
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index = data.index, columns = data.columns)
        


# In[150]:


indices.style.apply(highlight_max)







