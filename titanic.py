import pandas

import sklearn
from sklearn import tree, preprocessing, ensemble
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import accuracy_score

import tensorflow
from tensorflow.contrib import learn

# enable logging for debugging
tensorflow.logging.set_verbosity(tensorflow.logging.INFO)

# load data
data = pandas.read_csv("data.csv")
data = data.dropna()

# replace null data with 0
y, X = data['Survived'], data[['Age','SibSp', 'Fare']].fillna(0)

# split training data and test data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=123)

# train model using Decision Tree Classifier
dt_classifier = tree.DecisionTreeClassifier(max_depth=3) 
dt_classifier.fit(X_train, y_train)

# train model using Deep Neural Network using TensorFlow
dnn_classifier = learn.DNNClassifier(hidden_units=[20, 40, 20],
		n_classes=2,
		)
dnn_classifier.fit(X_train, y_train, steps = 1000)
dnn_prediction = dnn_classifier.predict(X_test)

# train model using Random Forest
rf_classifier = ensemble.RandomForestClassifier(n_estimators=50)
rf_classifier.fit(X_train, y_train)

print('Decision Tree Prediction Score: {0}'.format(dt_classifier.score(X_test, y_test)))
print('DNN Prediction Score: {0}'.format( accuracy_score(dnn_prediction, y_test)))
print('Random Forest Prediction Score: {0}'.format(rf_classifier.score(X_test, y_test)))
