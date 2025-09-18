from Part1a.machine_learning import neural_network
from Part1a.machine_learning import preprocess_data
from sklearn.feature_extraction.text import CountVectorizer
import pandas


def agent():

    state = "START"
    user_input = ""

    while True:
        state, message = state_transaction_function(state, user_input)

        user_input = input(f"{message}\n").lower()


def state_transaction_function(state, user_input):
    if state == "start":
        return "introduction", "Hello! welcome to restaurant search engine how can I help you?"

    if state == "introduction":
        vectorized = vectorizer.transform([user_input])
        prediction = model.predict(vectorized)

        print(prediction[0])

        if prediction == "inform":
            return "ask_area", "in what area you looking for a restaurant"



    if state == "ask_area":
        return "check_area", f"You are looking for a restaurant in {user_input}?"


    #return "INTRODUCTION", "Hello! welcome to restaurant search engine how can I help you?"


    # elif state == "INFORM":

    # else:
    #     return "INTRODUCTION", "Hello! welcome to restaurant search engine how can I help you?"


if __name__ == "__main__":
    vectorizer = CountVectorizer(max_features=3000)

    train_data = pandas.read_csv("../train_data.csv")
    test_data = pandas.read_csv("../test_data.csv")

    train_data_no_duplicate = train_data.drop_duplicates()
    test_data_no_duplicate = test_data.drop_duplicates()

    x_train, y_train, x_test, y_test = preprocess_data(vectorizer, train_data_no_duplicate, test_data_no_duplicate)

    model = neural_network(x_train, y_train, x_test, y_test)

    agent()