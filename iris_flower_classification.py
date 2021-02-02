#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#Load Dataset
url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names=['sepal-length','sepal-width','petal-length','petal-width','class']
dataset=read_csv(url,names=names)


# In[2]:


#summarize data
print(dataset.shape)
#head
print(dataset.head(20))
#descriptions
print(dataset.describe())
#class distribution
print(dataset.groupby('class').size())


# In[3]:


#data visualization

#box and whisker plots
dataset.plot(kind='box',subplots=True,layout=(2,2),sharey=False)
pyplot.show()

dataset.hist()
pyplot.show()


# In[4]:


#multivariant plots

#scatter plot matrix
scatter_matrix(dataset)
pyplot.show()


# In[30]:


#split out the validation data
array=dataset.values
x=array[:,0:4]
y=array[:,4]
x_train, x_validation, y_train, y_validation =train_test_split(x,y,test_size=0.2,random_state=1)


# In[31]:


#spot check algorithms

models=[]
models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))


# In[32]:


#Evaluate each model in turn
results=[]
names=[]
for name,model in models:
    kfold=StratifiedKFold(n_splits=10,random_state=1,shuffle=True)
    cv_results=cross_val_score(model,x_train,y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s,%f (%f)' %(name,cv_results.mean(),cv_results.std()))


# In[33]:


#compare algorithms
pyplot.boxplot(results,labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()


# In[34]:


#predictions
model=SVC(gamma='auto')
model.fit(x_train,y_train)
predictions=model.predict(x_validation)


# In[36]:


#evaluate predictions
print(accuracy_score(y_validation,predictions))
print(confusion_matrix(y_validation,predictions))
print(classification_report(y_validation,predictions))


# In[ ]:




