"""

    GRIP - The Spark Foundation

    Task:-3 Prediction using Decision Tree Algorithm

    Name:- Divya Patel

"""

# Importing libraries
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
from sklearn.tree import plot_tree

# Loading iris dataset
dataset=pd.read_csv('E:/Divya/datasets/Iris.csv')
x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

#Labeling Data
label=LabelEncoder()
y=label.fit_transform(y)

#spliting Data into train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

#Defining Decision Tree Classifier
data=DecisionTreeClassifier()
data.fit(x_train,y_train)

#Prediction on Test set
y_pred=data.predict(x_test)
print('y_pred:- ',y_pred)

#Finding Accuracy of Data
acc_dt=accuracy_score(y_test,y_pred)
print('accuracy:- ',acc_dt)

#Confusion merix
cm_dt=confusion_matrix(y_test,y_pred)
print('Confusion metrix:- \n',cm_dt)

#Showing Train Data on Scatter plot
x_set,y_set=x_train,y_train
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,2],color=ListedColormap(('red','blue','green'))(i),label=j)
plt.title('Decision Tree Classifier(Train Set)')
plt.xlabel('SepalLengthCm')
plt.ylabel('PetalLengthCm')
plt.legend()
plt.show()

#Showing Test Data on Scatter plot
x_set,y_set=x_test,y_test
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,2],color=ListedColormap(('red','blue','green'))(i),label=j)
plt.title('Decision Tree Classifier(Test Set)')
plt.xlabel('SepalLengthCm')
plt.ylabel('PetalLengthCm')
plt.legend()
plt.show()

#Showing Decision Tree
plt.figure(figsize=(12,7))
plot_tree(data,filled=True,feature_names=dataset.iloc[:,1:-1].columns)
plt.show()