import pandas as pd
hr=pd.read_csv("https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/HR.csv")
hr

print(hr.shape)

print(hr.keys())

print(hr.dtypes)

print(fruits['sales'].unique())
print(fruits['salary'].unique())

print(fruits.groupby('sales').size())
print(fruits.groupby('salary').size())

import seaborn as sns
import matplotlib.pyplot as plt
fig,ax=plt.subplots()
fig.set_size_inches(10,10)
sns.countplot(hr['sales'],label="Count")
plt.show()


print(hr.describe())

fruits.drop('left', axis=1).plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(10,10), title='Box Plot for each input variable')
plt.savefig('hr_sales')
plt.show()


feature_names =['satisfaction_level', 'last_evaluation','average_montly_hours', 'time_spend_company', 'Work_accident', 'left','promotion_last_5years']
X = fruits[feature_names]
y = fruits['number_project']

print(X.head())
print(y.head())


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape)
print(y_train.shape)


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))


from sklearn.naive_bayes import GaussianNB
nb= GaussianNB()
nb.fit(X_train, y_train)
print('Accuracy of Naive Bayesian classifier on training set: {:.2f}'.format(nb.score(X_train, y_train)))
print('Accuracy of Naive Bayesian classifier on test set: {:.2f}'.format(nb.score(X_test, y_test)))


from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'.format(svm.score(X_test, y_test)))
