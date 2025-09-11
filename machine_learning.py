from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas


def preprocess_data(train, test):
    # Load the pre-processed testing data
    train = train.dropna(subset=["utterance", "dialog_act"])
    test = test.dropna(subset=["utterance", "dialog_act"])

    x_train = vectorizer.fit_transform(train["utterance"])
    y_train = train["dialog_act"]

    x_test = vectorizer.transform(test["utterance"])
    y_test = test["dialog_act"]

    return x_train, y_train, x_test, y_test


def logistic_regression(x_train, y_train, x_test, y_test):
    model1 = LogisticRegression(max_iter=1000)
    model1.fit(x_train, y_train)

    y_prediction = model1.predict(x_test)

    accuracy = accuracy_score(y_test, y_prediction)
    print("Accuracy of logistic regression is:", accuracy)\

    return model1


def svm(x_train, y_train, x_test, y_test):
    model2 = SVC(kernel="rbf")
    model2.fit(x_train, y_train)

    y_prediction = model2.predict(x_test)

    accuracy = accuracy_score(y_test, y_prediction)
    print("Accuracy of SVM is:", accuracy)

    return model2


def neural_network(x_train, y_train, x_test, y_test):
    model3 = MLPClassifier()
    model3.fit(x_train, y_train)

    y_prediction = model3.predict(x_test)

    accuracy = accuracy_score(y_test, y_prediction)
    print("Accuracy of neural network is:", accuracy)

    return model3


vectorizer = CountVectorizer(max_features=3000)
train_data = pandas.read_csv("train_data.csv")
test_data = pandas.read_csv("test_data.csv")

x_train, y_train, x_test, y_test = preprocess_data(train_data, test_data)

logistic_regression(x_train, y_train, x_test, y_test)
svm(x_train, y_train, x_test, y_test)
neural_network(x_train, y_train, x_test, y_test)

# Now without duplicates
print("Now with the duplicates removed")
print("===================================================\n")


train_data_no_duplicate = train_data.drop_duplicates()
test_data_no_duplicate = test_data.drop_duplicates()

x_train, y_train, x_test, y_test = preprocess_data(train_data_no_duplicate, test_data_no_duplicate)

model1 = logistic_regression(x_train, y_train, x_test, y_test)
svm(x_train, y_train, x_test, y_test)
neural_network(x_train, y_train, x_test, y_test)

while True:
    sentence = input("what sentence do you want to classify?\n")

    vectorize = vectorizer.transform([sentence])

    prediction = model1.predict(vectorize)

    print("Predicted dialog_act:", prediction[0])