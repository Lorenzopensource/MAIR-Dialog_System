from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
import pandas
import random
import joblib


def preprocess_data(vectorizer, train, test):
    # Load the pre-processed testing data
    train = train.dropna(subset=["utterance", "dialog_act"])
    test = test.dropna(subset=["utterance", "dialog_act"])

    x_train = vectorizer.fit_transform(train["utterance"])
    y_train = train["dialog_act"]

    x_test = vectorizer.transform(test["utterance"])
    y_test = test["dialog_act"]

    return x_train, y_train, x_test, y_test

def logistic_regression(x_train, y_train, x_test, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    return model

def svm(x_train, y_train, x_test, y_test):
    model = SVC(kernel="rbf")
    model.fit(x_train, y_train)
    return model

def neural_network(x_train, y_train, x_test, y_test):
    model = MLPClassifier()
    model.fit(x_train, y_train)
    return model

def evaluation(model_name,model, x_test, y_test):
    print(f"Evaluation metrics for {model_name}:")
    print("Accuracy:", model.score(x_test, y_test))
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred, zero_division=1))

def error_analysis(x_test, y_test, model1, model2, model3, test_data):

    y_pred1 = model1.predict(x_test)
    y_pred2 = model2.predict(x_test)
    y_pred3 = model3.predict(x_test)
    y_pred4 = [rule_based_prediction(u) for u in test_data["utterance"]]

    # Counters of wrong predictions per dialog_act
    errors1, errors2, errors3, errors4 = {}, {}, {}, {}

    # List of utterances misclassified by all systems
    common_errors = []

    for i, true_label in enumerate(y_test):
        pred1, pred2, pred3, pred4 = y_pred1[i], y_pred2[i], y_pred3[i], y_pred4[i]

        if pred1 != true_label:
            errors1[true_label] = errors1.get(true_label, 0) + 1
        if pred2 != true_label:
            errors2[true_label] = errors2.get(true_label, 0) + 1
        if pred3 != true_label:
            errors3[true_label] = errors3.get(true_label, 0) + 1
        if pred4 != true_label:
            errors4[true_label] = errors4.get(true_label, 0) + 1

        if (pred1 != true_label and pred2 != true_label and 
            pred3 != true_label and pred4 != true_label):
            utterance = test_data.iloc[i]['utterance']
            common_errors.append((utterance, true_label, pred1, pred2, pred3, pred4))

    def print_sorted_errors(errors, name):
        print(f"\n--- {name} ---")
        sorted_errors = sorted(errors.items(), key=lambda x: x[1], reverse=True)
        for act, count in sorted_errors:
            print(f"{act}: {count}")

    print_sorted_errors(errors1, "Logistic Regression most frequent misclassified dialog_acts")
    print_sorted_errors(errors2, "SVM most frequent misclassified dialog_acts")
    print_sorted_errors(errors3, "Neural Network most frequent misclassified dialog_acts")
    print_sorted_errors(errors4, "Rule-Based (Baseline2) most frequent misclassified dialog_acts")

    print("\n===================================================")
    print(f"Number of sentences misclassified by ALL systems: {len(common_errors)}")

    # Saving common errors to file

    output_file = "misclassified_by_all.txt"
    if common_errors:
        with open(output_file, "w", encoding="utf-8") as f:
            for utterance, true, p1, p2, p3, p4 in common_errors:
                f.write(f"Utterance: {utterance}\n")
                f.write(f"True label: {true}\n")
                f.write(f"Logistic Regression predicted: {p1}\n")
                f.write(f"SVM predicted: {p2}\n")
                f.write(f"Neural Network predicted: {p3}\n")
                f.write(f"Rule-Based predicted: {p4}\n")
                f.write("---------------------------------------------------\n")

        print(f"\nAll common misclassifications exported to {output_file}")

if __name__ == "__main__":

    from baselines import rule_based_prediction
    from baselines import most_frequent_class_baseline


    vectorizer = CountVectorizer(max_features=3000)
    train_data = pandas.read_csv("train_data.csv")
    test_data = pandas.read_csv("test_data.csv")

    x_train, y_train, x_test, y_test = preprocess_data(vectorizer, train_data, test_data)

    dup_log_reg = logistic_regression(x_train, y_train, x_test, y_test)
    dup_svm= svm(x_train, y_train, x_test, y_test)
    dup_neural_net= neural_network(x_train, y_train, x_test, y_test)


    train_data_no_duplicate = train_data.drop_duplicates()
    test_data_no_duplicate = test_data.drop_duplicates()

    x_train, y_train, x_test, y_test = preprocess_data(vectorizer,train_data_no_duplicate, test_data_no_duplicate)

    log_reg = logistic_regression(x_train, y_train, x_test, y_test)
    svm_ = svm(x_train, y_train, x_test, y_test)
    neural_net = neural_network(x_train, y_train, x_test, y_test)

    #Saving the best model
    joblib.dump(log_reg, 'Utterance_Classifier_NN.pkl')
    joblib.dump(vectorizer, 'Vectorizer_NN.pkl')

    while True:
        user_input = input("Select the model to test or type 'evaluation' (type 'q' to quit) -  Model options: \n 'most_frequent' - 'rule_based' - 'logistic_regression' - 'svm' - 'neural network' \n  ") 

        if user_input.lower() == 'q':
            break
        if user_input.lower() == 'evaluation':
            evaluation("Logistic Regression with duplicates", dup_log_reg, x_test, y_test)
            evaluation("SVM with duplicates", dup_svm, x_test, y_test)
            evaluation("Neural Network with duplicates", dup_neural_net, x_test, y_test)
            evaluation("Logistic Regression without duplicates", log_reg, x_test, y_test)
            evaluation("SVM without duplicates", svm_, x_test, y_test)
            evaluation("Neural Network without duplicates", neural_net, x_test, y_test)
            error_analysis(x_test, y_test, log_reg, svm_, neural_net, test_data_no_duplicate)
            continue
        if user_input.lower() == 'most_frequent':
            user_input = input("Enter an utterance to classify (or 'q' to quit): ")
            if user_input.lower() == 'q':
                break
            print(f"Predicted Dialog Act: {most_frequent_class_baseline(train_data_path='train_data.csv')}")
            continue
        elif user_input.lower() == 'rule_based':
            user_input = input("Enter an utterance to classify (or 'q' to quit): ")
            if user_input.lower() == 'q':
                break
            predicted_act = rule_based_prediction(user_input)
            print(f"Predicted Dialog Act: {predicted_act}")
            continue
        elif user_input.lower() == 'logistic_regression':   
            user_input = input("Enter an utterance (or type 'q' to quit): ")
            if user_input.lower() == 'q':
                break
            vectorize = vectorizer.transform([user_input])
            prediction = log_reg.predict(vectorize)
            print("Predicted dialog_act:", prediction[0])
            continue
        elif user_input.lower() == 'svm':   
            user_input = input("Enter an utterance (or type 'q' to quit): ")
            if user_input.lower() == 'q':
                break
            vectorize = vectorizer.transform([user_input])
            prediction = svm_.predict(vectorize)
            print("Predicted dialog_act:", prediction[0])
            continue
        elif user_input.lower() == 'neural_network':   
            user_input = input("Enter an utterance (or type 'q' to quit): ")
            if user_input.lower() == 'q':
                break
            vectorize = vectorizer.transform([user_input])
            prediction = neural_net.predict(vectorize)
            print("Predicted dialog_act:", prediction[0])
            continue
        else:
            print("Invalid option. Please choose 'most_frequent' or 'rule_based'.")
