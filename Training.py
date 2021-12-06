import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

df = pd.read_csv('mx_covid_data.csv')
df = df.drop(['REGISTRATION ID'], axis=1)

# Data normalization
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

X = df[['PATIENT_TYPE','INTUBATED','PNEUMONIA','AGE','DIABETES','COPD','ASTHMA','INMUSUPR','HYPERTENSION','OTRA_COM','CARDIOVASCULAR','OBESITY','KIDNEY CHRONIC','SMOKING','ICU','SEX_PREGNANCY']]
y = df['DIE_SURVIVAL']

# training, test sets - ratio is 8:2.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Use cross-validation to select maximum depth
mean = []
standard_deviation = []
for i in range(1,30):
    # train decision tree model
    model = DecisionTreeClassifier(criterion='entropy', max_depth=i, random_state=1)
    scores = cross_val_score(model,X_train,y_train,cv=5,scoring='f1')
    mean.append(np.array(scores).mean())
    standard_deviation.append(np.array(scores).std())
    # make prediction
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

# plot the cross-validation plot
plt.errorbar(range(1,30),mean,yerr=standard_deviation)
plt.xlabel('max_depth')
plt.ylabel('F1 Score')
plt.show()

# Use cross-validation to select k for kNN classifier
mean = []
standard_deviation = []
for k in range(1,20):
    # train kNN classifier
    model = KNeighborsClassifier(n_neighbors=k,weights='uniform')
    scores = cross_val_score(model,X_train,y_train,cv=5,scoring='f1')
    mean.append(np.array(scores).mean())
    standard_deviation.append(np.array(scores).std())
    # make prediction
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
# plot the cross-validation plot
plt.errorbar(range(1,20),mean,yerr=standard_deviation)
plt.xlabel('k')
plt.ylabel('F1 Score')
plt.show()

# Train decision tree model with max_depth=4
tree = DecisionTreeClassifier(criterion = 'entropy',max_depth=4, random_state=1)
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)
print('Decision Tree Accuracy: ', metrics.accuracy_score(y_test, tree_pred))

# confusion matrix of decision tree model
print('Decision Tree Confusion Matrix')
print(confusion_matrix(y_test, tree_pred))
print(metrics.classification_report(y_test, tree_pred))

# Train Gaussian Naive Bayes classifier
GNB = GaussianNB()
GNB.fit(X_train, y_train)
GNB_pred = GNB.predict(X_test)
print('Gaussian Naive Bayes Classifier Accuracy: ', metrics.accuracy_score(y_test, GNB_pred))

# confusion matrix of logistic regression model
print('Gaussian Naive Bayes Classifier Confusion Matrix')
print(confusion_matrix(y_test, GNB_pred))
print(metrics.classification_report(y_test, GNB_pred))

# Training knn model
knn = KNeighborsClassifier(n_neighbors=5,weights='uniform')
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print('kNN Accuracy: ', metrics.accuracy_score(y_test, knn_pred))

# confusion matrix of kNN
print('kNN Confusion Matrix')
print(confusion_matrix(y_test, knn_pred))
print(metrics.classification_report(y_test, knn_pred))

# baseline classifier (most frequent class in training data)
# train dummy classifier
dummy1 = DummyClassifier(strategy='most_frequent')
dummy1.fit(X_train,y_train)
dummy1_pred = dummy1.predict(X_test)
print('Dummy Classifier (Most Frequent Class) Accuracy: ', metrics.accuracy_score(y_test, dummy1_pred))
# confusion matrix of dummy classifier
print('Dummy Classifier (Most Frequent Class) Confusion Matrix')
print(confusion_matrix(y_test,dummy1_pred))
print(metrics.classification_report(y_test, dummy1_pred))

# baseline classifier (Uniform random prediction)
# train dummy classifier
dummy2 = DummyClassifier(strategy='uniform')
dummy2.fit(X_train,y_train)
dummy2_pred = dummy2.predict(X_test)
print('Dummy Classifier (Random Prediction) Accuracy: ', metrics.accuracy_score(y_test, dummy2_pred))
# confusion matrix of dummy classifier
print('Dummy Classifier (Random Prediction) Confusion Matrix')
print(confusion_matrix(y_test,dummy2_pred))
print(metrics.classification_report(y_test, dummy2_pred))

# plot ROC curves
plt.rc('font',size=18)
plt.rcParams['figure.constrained_layout.use']=True
# decision tree
fpr,tpr,_ = roc_curve(y_test,np.array(tree.predict_proba(X_test).tolist())[:,1])
plt.plot(fpr,tpr,color='green')
# Gaussian Naive Bayes Classifier
fpr,tpr,_ = roc_curve(y_test,np.array(GNB.predict_proba(X_test).tolist())[:,1])
plt.plot(fpr,tpr,color='pink')
# knn
fpr,tpr,_ = roc_curve(y_test,np.array(knn.predict_proba(X_test).tolist())[:,1])
plt.plot(fpr,tpr,color='yellow')
# baseline(most frequent)
fpr,tpr,_ = roc_curve(y_test,np.array(dummy1.predict_proba(X_test).tolist())[:,1])
plt.plot(fpr,tpr,color='red')
# baseline(random prediction)
fpr,tpr,_ = roc_curve(y_test,np.array(dummy2.predict_proba(X_test).tolist())[:,1])
plt.plot(fpr,tpr,color='blue')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(['Decision Tree','GaussianNB','kNN','Baseline(most frequent)','Baseline(random prediction)'])
plt.plot([0,1],[0,1],color='black',linestyle='-.')
plt.show()

# Plot heat map to view the correlation of features
corr_mat = df.corr()

plt.figure(figsize=(20,20))
g=sns.heatmap(corr_mat,annot=True,cmap="RdYlGn")
plt.show()

X1 = df[['PATIENT_TYPE','INTUBATED','PNEUMONIA','AGE']]
y1 = df['DIE_SURVIVAL']

# training, test sets - ratio is 8:2.
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=1)

# Train decision tree model with max_depth=4
tree1 = DecisionTreeClassifier(criterion = 'entropy',max_depth=4, random_state=1)
tree1.fit(X_train1, y_train1)
tree_pred1 = tree1.predict(X_test1)
print('Decision Tree Accuracy: ', metrics.accuracy_score(y_test1, tree_pred1))

# confusion matrix of decision tree model
print('Decision Tree Confusion Matrix')
print(confusion_matrix(y_test1, tree_pred1))
print(metrics.classification_report(y_test1, tree_pred1))

# Train Gaussian Naive Bayes
GNB1 = GaussianNB()
GNB1.fit(X_train1, y_train1)
GNB_pred1 = GNB1.predict(X_test1)
print('Gaussian Naive Bayes Classifier Accuracy: ', metrics.accuracy_score(y_test1, GNB_pred1))

# confusion matrix of Gaussian Naive Bayes
print('Gaussian Naive Bayes Classifier Confusion Matrix')
print(confusion_matrix(y_test1, GNB_pred1))
print(metrics.classification_report(y_test1, GNB_pred1))

# Train kNN
knn1 = KNeighborsClassifier(n_neighbors=5,weights='uniform')
knn1.fit(X_train1, y_train1)
knn_pred1 = knn1.predict(X_test1)
print('kNN Accuracy: ', metrics.accuracy_score(y_test1, knn_pred1))

# confusion matrix of kNN
print('kNN Confusion Matrix')
print(confusion_matrix(y_test1, knn_pred1))
print(metrics.classification_report(y_test1, knn_pred1))

# baseline classifier (most frequent class in training data)
# train dummy classifier
dummy3 = DummyClassifier(strategy='most_frequent')
dummy3.fit(X_train1,y_train1)
dummy3_pred = dummy3.predict(X_test1)
print('Dummy Classifier (Most Frequent Class) Accuracy: ', metrics.accuracy_score(y_test1, dummy3_pred))
# confusion matrix of dummy classifier
print('Dummy Classifier (Most Frequent Class) Confusion Matrix')
print(confusion_matrix(y_test1,dummy3_pred))
print(metrics.classification_report(y_test1, dummy3_pred))

# baseline classifier (Uniform random prediction)
# train dummy classifier
dummy4 = DummyClassifier(strategy='uniform')
dummy4.fit(X_train1,y_train1)
dummy4_pred = dummy4.predict(X_test1)
print('Dummy Classifier (Random Prediction) Accuracy: ', metrics.accuracy_score(y_test1, dummy4_pred))
# confusion matrix of dummy classifier
print('Dummy Classifier (Random Prediction) Confusion Matrix')
print(confusion_matrix(y_test1,dummy4_pred))
print(metrics.classification_report(y_test1, dummy4_pred))

# plot ROC curves
plt.rc('font',size=18)
plt.rcParams['figure.constrained_layout.use']=True
# decision tree
fpr,tpr,_ = roc_curve(y_test1,np.array(tree1.predict_proba(X_test1).tolist())[:,1])
plt.plot(fpr,tpr,color='green')
# Gaussian Naive Bayes Classifier
fpr,tpr,_ = roc_curve(y_test1,np.array(GNB1.predict_proba(X_test1).tolist())[:,1])
plt.plot(fpr,tpr,color='pink')
# knn
fpr,tpr,_ = roc_curve(y_test1,np.array(knn1.predict_proba(X_test1).tolist())[:,1])
plt.plot(fpr,tpr,color='yellow')
# baseline(most frequent)
fpr,tpr,_ = roc_curve(y_test1,np.array(dummy3.predict_proba(X_test1).tolist())[:,1])
plt.plot(fpr,tpr,color='red')
# baseline(random prediction)
fpr,tpr,_ = roc_curve(y_test1,np.array(dummy4.predict_proba(X_test1).tolist())[:,1])
plt.plot(fpr,tpr,color='blue')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(['Decision Tree','GaussianNB','kNN','Baseline(most frequent)','Baseline(random prediction)'])
plt.plot([0,1],[0,1],color='black',linestyle='-.')
plt.show()