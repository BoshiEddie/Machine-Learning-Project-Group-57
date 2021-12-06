import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyClassifier

from sklearn.feature_selection import SelectKBest,chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier, GradientBoostingClassifier

df = pd.read_csv('mx_covid_data.csv')
df = df.drop(['REGISTRATION ID'], axis=1)

# Data normalization
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

X = df[['PATIENT_TYPE','INTUBATED','PNEUMONIA','AGE','DIABETES','COPD','ASTHMA','INMUSUPR','HYPERTENSION','OTRA_COM','CARDIOVASCULAR','OBESITY','KIDNEY CHRONIC','SMOKING','ICU','SEX_PREGNANCY']]
y = df['DIE_SURVIVAL']

# training, test sets - ratio is 8:2.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# # Use cross-validation to select k for kNN classifier
# mean = []
# standard_deviation = []
# for k in range(1,20):
#     # train kNN classifier
#     model = KNeighborsClassifier(n_neighbors=k,weights='uniform')
#     scores = cross_val_score(model,X_train,y_train,cv=5,scoring='f1')
#     mean.append(np.array(scores).mean())
#     standard_deviation.append(np.array(scores).std())
#     # make prediction
#     model.fit(X_train,y_train)
#     y_pred = model.predict(X_test)
# # plot the cross-validation plot
# plt.errorbar(range(1,20),mean,yerr=standard_deviation)
# plt.xlabel('k')
# plt.ylabel('F1 Score')
# plt.show()

knn = KNeighborsClassifier(n_neighbors=5,weights='uniform')
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print('kNN Accuracy: ', metrics.accuracy_score(y_test, knn_pred))

# confusion matrix of kNN
print('kNN Confusion Matrix')
print(confusion_matrix(y_test, knn_pred))

# baseline classifier (most frequent class in training data)
# train dummy classifier
dummy1 = DummyClassifier(strategy='most_frequent')
dummy1.fit(X_train,y_train)
dummy1_pred = dummy1.predict(X_test)
print('Dummy Classifier (Most Frequent Class) Accuracy: ', metrics.accuracy_score(y_test, dummy1_pred))
# confusion matrix of dummy classifier
print('Dummy Classifier (Most Frequent Class) Confusion Matrix')
print(confusion_matrix(y_test,dummy1_pred))

# baseline classifier (Uniform random prediction)
# train dummy classifier
dummy2 = DummyClassifier(strategy='uniform')
dummy2.fit(X_train,y_train)
dummy2_pred = dummy2.predict(X_test)
print('Dummy Classifier (Random Prediction) Accuracy: ', metrics.accuracy_score(y_test, dummy2_pred))
# confusion matrix of dummy classifier
print('Dummy Classifier (Random Prediction) Confusion Matrix')
print(confusion_matrix(y_test,dummy2_pred))

# plot ROC curves
plt.rc('font',size=18)
plt.rcParams['figure.constrained_layout.use']=True
# Gaussian Naive Bayes Classifier
fpr,tpr,_ = roc_curve(y_test,np.array(knn.predict_proba(X_test).tolist())[:,1])
plt.plot(fpr,tpr,color='green')
# baseline(most frequent)
fpr,tpr,_ = roc_curve(y_test,np.array(dummy1.predict_proba(X_test).tolist())[:,1])
plt.plot(fpr,tpr,color='red')
# baseline(random prediction)
fpr,tpr,_ = roc_curve(y_test,np.array(dummy2.predict_proba(X_test).tolist())[:,1])
plt.plot(fpr,tpr,color='blue')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(['kNN','Baseline(most frequent)','Baseline(random prediction)'])
plt.plot([0,1],[0,1],color='black',linestyle='-.')
plt.show()


corr_mat = df.corr()

plt.figure(figsize=(20,20))
g=sns.heatmap(corr_mat,annot=True,cmap="RdYlGn")
plt.show()

X1 = df[['PATIENT_TYPE','INTUBATED','PNEUMONIA','AGE']]
y1 = df['DIE_SURVIVAL']

# training, test sets - ratio is 8:2.
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=1)

# Train kNN
knn1 = KNeighborsClassifier(n_neighbors=5,weights='uniform')
knn1.fit(X_train1, y_train1)
knn_pred1 = knn1.predict(X_test1)
print('kNN Accuracy: ', metrics.accuracy_score(y_test1, knn_pred1))

# confusion matrix of kNN
print('kNN Confusion Matrix')
print(confusion_matrix(y_test1, knn_pred1))

# baseline classifier (most frequent class in training data)
# train dummy classifier
dummy3 = DummyClassifier(strategy='most_frequent')
dummy3.fit(X_train1,y_train1)
dummy3_pred = dummy3.predict(X_test1)
print('Dummy Classifier (Most Frequent Class) Accuracy: ', metrics.accuracy_score(y_test1, dummy3_pred))
# confusion matrix of dummy classifier
print('Dummy Classifier (Most Frequent Class) Confusion Matrix')
print(confusion_matrix(y_test1,dummy3_pred))

# baseline classifier (Uniform random prediction)
# train dummy classifier
dummy4 = DummyClassifier(strategy='uniform')
dummy4.fit(X_train1,y_train1)
dummy4_pred = dummy4.predict(X_test1)
print('Dummy Classifier (Random Prediction) Accuracy: ', metrics.accuracy_score(y_test1, dummy4_pred))
# confusion matrix of dummy classifier
print('Dummy Classifier (Random Prediction) Confusion Matrix')
print(confusion_matrix(y_test1,dummy4_pred))

# plot ROC curves
plt.rc('font',size=18)
plt.rcParams['figure.constrained_layout.use']=True
# kNN
fpr,tpr,_ = roc_curve(y_test1,np.array(knn1.predict_proba(X_test1).tolist())[:,1])
plt.plot(fpr,tpr,color='green')
# baseline(most frequent)
fpr,tpr,_ = roc_curve(y_test1,np.array(dummy3.predict_proba(X_test1).tolist())[:,1])
plt.plot(fpr,tpr,color='red')
# baseline(random prediction)
fpr,tpr,_ = roc_curve(y_test1,np.array(dummy4.predict_proba(X_test1).tolist())[:,1])
plt.plot(fpr,tpr,color='blue')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(['kNN','Baseline(most frequent)','Baseline(random prediction)'])
plt.plot([0,1],[0,1],color='black',linestyle='-.')
plt.show()