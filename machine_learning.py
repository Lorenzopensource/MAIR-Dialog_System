from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas


def preprocess_data():
    # Load the pre-processed testing data
    train_data = pandas.read_csv("train_data.csv")
    test_data = pandas.read_csv("test_data.csv")

    train_data = train_data.dropna(subset=["utterance", "dialog_act"])
    test_data = test_data.dropna(subset=["utterance", "dialog_act"])

    vectorizer = CountVectorizer(max_features=3000)

    x_train = vectorizer.fit_transform(train_data["utterance"])
    y_train = train_data["dialog_act"]

    x_test = vectorizer.transform(test_data["utterance"])
    y_test = test_data["dialog_act"]

    return x_train, y_train, x_test, y_test


def logistic_regression():
    model1 = LogisticRegression(max_iter=1000)
    model1.fit(x_train, y_train)

    y_prediction = model1.predict(x_test)

    accuracy = accuracy_score(y_test, y_prediction)
    print("Accuracy of logistic regression is:", accuracy)


def svm():
    model2 = SVC(kernel="rbf")
    model2.fit(x_train, y_train)

    y_prediction = model2.predict(x_test)

    accuracy = accuracy_score(y_test, y_prediction)
    print("Accuracy of SVM is:", accuracy)


def neural_network():
    model3 = MLPClassifier()
    model3.fit(x_train, y_train)

    y_prediction = model3.predict(x_test)

    accuracy = accuracy_score(y_test, y_prediction)
    print("Accuracy of neural network is:", accuracy)


x_train, y_train, x_test, y_test = preprocess_data()

logistic_regression()
svm()
neural_network()
