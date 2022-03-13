



#import the dependencies
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report , confusion_matrix
from sklearn.metrics import accuracy_score

# loading the dataset to a pandas dataframe

credit_card_data = pd.read_csv('creditcard.csv')


#first 5 rows of the dataset
credit_card_data.head()

#last 5 rows of the dataset
credit_card_data.tail()


#data information 
#credit_card_data.info()

#checking null values present in each column
#credit_card_data.isnull().sum()

#distributions of legit transaction and fraudulent transaction
credit_card_data['Class'].value_counts()

##THIS IS AN HIGHLY IMBALANCE DATASET.
#WE ARE GOING TO PREPROCESS IT FOR OUR FURTHUR USE.
# 0------> NORMAL TRANSACTION
# 1------> FRAUDULENT TRANSACTION
#separating the data for analysis

legit= credit_card_data[credit_card_data.Class == 0]
fraud= credit_card_data[credit_card_data.Class == 1]

print(legit.shape)
print(fraud.shape)

#statistical measure of the data
legit.Amount.describe()

fraud.Amount.describe()

#compare the values for both transaction
credit_card_data.groupby('Class').mean()

#we are going to create sample dataset from this unbalanced data set
#using undersampling
legit_sample= legit.sample(n=492)#because the fraud transactions are also 492.
#now we have to concatinate the above two dataframes.

new_dataset= pd.concat([legit_sample,fraud], axis=0)
#to see the first 5 rows;
new_dataset.head()

#last five rows
new_dataset.tail()

#checking the no of legit and fraud transactions on this new dataset
new_dataset['Class'].value_counts()
new_dataset.groupby('Class').mean()
#splitting the dataset into features and target
X=new_dataset.drop(columns = 'Class', axis=1)
Y=new_dataset['Class']
#print(X)
#print(Y)

#split the data into training and testing data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2)
#print(X.shape,X_train.shape,X_test.shape)

#model training
#using logistic regression

model=LogisticRegression()

#training the logistic regression model with training data

model.fit(X_train, Y_train)


#model evaluation

#accuracy score on training data
X_train_prediction= model.predict(X_train)
training_data_accuracy= accuracy_score(X_train_prediction, Y_train)


print('Accuracy On Training Data for logistic regression',training_data_accuracy)

#accuracy score on test data
X_test_prediction= model.predict(X_test)
test_data_accuracy= accuracy_score(X_test_prediction, Y_test)

print('Accuracy On test Data for logistic regression',test_data_accuracy)

#second model training
#using k-nearest neighbors algorithm

classifier= KNeighborsClassifier(9)

#training the knn model with training data
classifier.fit(X_train, Y_train)

#second model evaluation
y_prediction=classifier.predict(X_test)

print(classification_report(Y_test,y_prediction))
print(confusion_matrix(Y_test,y_prediction))

print('Accuracy_score for KNN ',accuracy_score(Y_test,y_prediction))




# In[4]:


credit_card1 = pd.read_csv('test.csv')


# In[5]:


credit_card1.head()


# In[6]:


Z=credit_card1.drop(columns = 'Class', axis=1)


# In[7]:


W=credit_card1['Class']


# In[8]:


#print(W)


# In[9]:


Z_test=Z


# In[18]:


W_test=W


# In[19]:


#model1=LogisticRegression()
#model1.fit(X_train,Y_train)
classifier1= KNeighborsClassifier(9)
classifier1.fit(X_train, Y_train)
#Z_test_prediction= model1.predict(Z_test)
Z_test_prediction= classifier1.predict(Z_test)
test1_data_accuracy= accuracy_score(Z_test_prediction,W_test)
print('the result of the test is:->',Z_test_prediction)
if Z_test_prediction == 1:
    print('fraud')
else:
    print('normal')

print('Accuracy On Data for single transaction',test1_data_accuracy)


# In[20]:


credit_card2 = pd.read_csv('test2.csv')
credit_card2.head()


# In[21]:


R=credit_card2.drop(columns = 'Class', axis=1)
S=credit_card2['Class']
#print(S)


# In[25]:


R_test=R
S_test=S


# In[26]:


model2=LogisticRegression()
model2.fit(X_train,Y_train)

R_test_prediction= model2.predict(R_test)
test2_data_accuracy= accuracy_score(R_test_prediction,S_test)
print('The result of the test for logistic regression  is:->',R_test_prediction)
print('Accuracy On Data for 5 transaction',test2_data_accuracy)
#if Z_test_prediction == 1:
   # print('fraud')
#else:
    #print('normal')



# In[27]:


classifier2= KNeighborsClassifier(9)
classifier2.fit(X_train, Y_train)
R_test_prediction= classifier2.predict(R_test)
test2_data_accuracy= accuracy_score(R_test_prediction,S_test)
print('The result of the test for KNN is:->',R_test_prediction)
print('Accuracy On Data for 5 transaction',test2_data_accuracy)


# In[ ]:





# In[ ]:




