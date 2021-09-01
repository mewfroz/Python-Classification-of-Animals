import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

import os



from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import *
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score
from sklearn.model_selection import cross_val_score


zoo_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'zoo.csv'))
class_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'class.csv'))
zoo_df.head()
class_df.head()


animal_df = zoo_df.merge(class_df,how='left',left_on='class_type',right_on='Class_Number')
animal_df.head()


zoo_df = animal_df.drop(['class_type','Animal_Names', 'Number_Of_Animal_Species_In_Class'], axis=1)
zoo_df.head()

zoo_df.isnull().any()


zoo_df.info()

zoo_df.describe()


sns.set_style('whitegrid')
print(zoo_df)


plt.rcParams['figure.figsize'] = (7,7)
sns.countplot(zoo_df['Class_Type'], palette='YlGnBu')
ax = plt.gca()
ax.set_title("Histogram of Classes")
plt.show()

zoo_df['has_legs'] = np.where(zoo_df['legs']>0,1,0)
zoo_df = zoo_df[['animal_name','hair','feathers','eggs','milk', 'airborne', 'aquatic', 'predator', 'toothed', 'backbone', 'breathes','venomous','fins','legs','has_legs','tail','domestic','catsize','Class_Number','Class_Type']]
zoo_df.head()

zoo_df_temp = zoo_df.drop(['has_legs','Class_Number'], axis=1)
zoo_df_temp = zoo_df_temp.groupby(by='animal_name').mean()
plt.rcParams['figure.figsize'] = (16,10) 
sns.heatmap(zoo_df_temp, cmap="inferno")
ax = plt.gca()
ax.set_title("Features for the Animals")
plt.show()

zoo_df_temp = zoo_df.drop(['has_legs','Class_Number'], axis=1)
zoo_df_temp = zoo_df_temp.groupby(by='Class_Type').mean()
plt.rcParams['figure.figsize'] = (16,10) 
sns.heatmap(zoo_df_temp, annot=True, cmap="inferno")
ax = plt.gca()
ax.set_title("HeatMap of Features for the Classes")
plt.show()

zoo_df_temp = zoo_df.drop(['legs','Class_Number'], axis=1)
zoo_df_temp = zoo_df_temp.groupby(by='animal_name').mean()
plt.rcParams['figure.figsize'] = (16,10) 
sns.heatmap(zoo_df_temp, cmap="inferno")
ax = plt.gca()
ax.set_title("Features for the Animals")
plt.show()

zoo_df_temp = zoo_df.drop(['legs','Class_Number'], axis=1)
zoo_df_temp = zoo_df_temp.groupby(by='Class_Type').mean()
plt.rcParams['figure.figsize'] = (16,10) 
sns.heatmap(zoo_df_temp, annot=True, cmap="inferno")
ax = plt.gca()
ax.set_title("HeatMap of Features for the Classes")
plt.show()

zoo_df.head()


features = list(zoo_df.columns.values)
features.remove('has_legs')
features.remove('Class_Type')
features.remove('Class_Number')
features.remove('animal_name')
X = zoo_df[features]
y = zoo_df['Class_Number']


X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 0)


knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)


y_pred = knn.predict(X_test)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

plt.rcParams['figure.figsize'] = (9,9) 
_, ax = plt.subplots()
ax.hist(y_test, color = 'm', alpha = 0.5, label = 'actual', bins=7)
ax.hist(y_pred, color = 'c', alpha = 0.5, label = 'prediction', bins=7)
ax.yaxis.set_ticks(np.arange(0,11))
ax.legend(loc = 'best')
plt.show()


k_list = np.arange(1, 50, 2)
mean_scores = []
accuracy_list = []
error_rate = []

for i in k_list:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    score = cross_val_score(knn,X_train, y_train,cv=10)
    mean_scores.append(np.mean(score))
    error_rate.append(np.mean(pred_i != y_test))

print("Mean Scores:")
print(mean_scores)
print("Error Rate:")
print(error_rate)


plt.plot(k_list,mean_scores, marker='o')


plt.title('Accuracy of Model for Varying Values of K')
plt.xlabel("Values of K")
plt.ylabel("Mean Accuracy Score")
plt.xticks(k_list)
plt.rcParams['figure.figsize'] = (12,12) 
plt.show()


plt.plot(k_list,error_rate, color='r', marker = 'o')


plt.title('Error Rate for Model for Varying Values of K')
plt.xlabel("Values of K")
plt.ylabel("Error Rate")
plt.xticks(k_list)
plt.rcParams['figure.figsize'] = (12,12) 
plt.show()


features = list(zoo_df.columns.values)
features.remove('legs')
features.remove('Class_Type')
features.remove('Class_Number')
features.remove('animal_name')
X2 = zoo_df[features]
y2 = zoo_df['Class_Type']


X2_train, X2_test, y2_train, y2_test = train_test_split(X2,y2,random_state = 0)


knn2 = KNeighborsClassifier(n_neighbors = 5)
knn2.fit(X2_train, y2_train)


y2_pred = knn2.predict(X2_test)

print(confusion_matrix(y2_test,y2_pred))

print(classification_report(y2_test,y2_pred))

plt.rcParams['figure.figsize'] = (9,9) 
_, ax = plt.subplots()
ax.hist(y2_test, color = 'm', alpha = 0.5, label = 'actual', bins=7)
ax.hist(y2_pred, color = 'c', alpha = 0.5, label = 'prediction', bins=7)
ax.yaxis.set_ticks(np.arange(0,11))
ax.legend(loc = 'best')
plt.show()


k_list = np.arange(1, 50, 2)
mean_scores2 = []
accuracy_list2 = []
error_rate2 = []

for i in k_list:
    knn2 = KNeighborsClassifier(n_neighbors=i)
    knn2.fit(X2_train,y2_train)
    pred_i = knn2.predict(X2_test)
    score = cross_val_score(knn2,X2_train, y2_train,cv=10)
    mean_scores2.append(np.mean(score))
    error_rate2.append(np.mean(pred_i != y2_test))

print("Mean Scores:")
print(mean_scores)
print("Error Rate:")
print(error_rate)


plt.plot(k_list,mean_scores, color='b',marker='o', label='Model using Number of Legs')
plt.plot(k_list,mean_scores2, color='m',marker='x', label='Model using Presence of Legs')


plt.title('Accuracy of Model for Varying Values of K')
plt.xlabel("Values of K")
plt.ylabel("Mean Accuracy Score")
plt.xticks(k_list)
plt.legend()
plt.rcParams['figure.figsize'] = (12,12) 
plt.show()


plt.plot(k_list,error_rate, color='r', marker = 'o', label='Model using Number of Legs')
plt.plot(k_list,error_rate2, color='c', marker = 'x', label='Model using Presence of Legs')


plt.title('Error Rate for Model for Varying Values of K')
plt.xlabel("Values of K")
plt.ylabel("Error Rate")
plt.xticks(k_list)
plt.legend()
plt.rcParams['figure.figsize'] = (12,12) 
plt.show()

