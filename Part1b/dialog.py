from Part1a.machine_learning import neural_network
from Part1a.machine_learning import preprocess_data
from sklearn.feature_extraction.text import CountVectorizer
import pandas


def agent():
    context = {"area": "", "food_type": "", "price": ""}
    state = "start"
    user_input = ""

    while True:
        state, message = state_transaction_function(state, user_input, context)

        user_input = input(f"{message}\n").lower()


def state_transaction_function(state, user_input, context):
    if state == "start":
        return "introduction", "Hello! welcome to restaurant search engine how can I help you?"

    if state == "introduction":
        vectorized = vectorizer.transform([user_input])
        prediction = model.predict(vectorized)

        if prediction[0] == "inform":
            return "ask_area", "in what area you looking for a restaurant"

    if state == "ask_area":
        context["area"] = user_input
        return "check_area", f"You are looking for a restaurant in {user_input}?"

    if state == "ask_foodtype":
        context["food_type"] = user_input
        return "check_foodtype", f"You are looking for {user_input} food?"

    if state == "ask_price":
        context["price"] = user_input
        return "check_price", f"You are looking for the price range {user_input}?"

    if state == "check_area":
        vectorized = vectorizer.transform([user_input])
        prediction = model.predict(vectorized)

        if prediction[0] == "affirm":
            return "ask_foodtype", "for what food type are you looking?"
        else:
            return "ask_area", "in what area you looking for a restaurant"


    if state == "check_foodtype":
        vectorized = vectorizer.transform([user_input])
        prediction = model.predict(vectorized)

        if prediction[0] == "affirm":
            return "ask_price", "In what price range are you looking?"
        else:
            return "ask_foodtype", "for what food type are you looking?"

    if state == "check_price":
        vectorized = vectorizer.transform([user_input])
        prediction = model.predict(vectorized)

        if prediction[0] == "affirm":
            return "confirmation", f"So you are looking for a restaurant in {context["area"]} with {context["food_type"]} food in the price range {context["price"]} right?"
        else:
            return "ask_foodtype", "for what food type are you looking?"

    if state == "confirmation":
        vectorized = vectorizer.transform([user_input])
        prediction = model.predict(vectorized)

        if prediction[0] == "affirm":
            print(context)
        else:
            return "ask_area", "in what area you looking for a restaurant"



if __name__ == "__main__":
    vectorizer = CountVectorizer(max_features=3000)

    train_data = pandas.read_csv("train_data.csv")
    test_data = pandas.read_csv("test_data.csv")

    train_data_no_duplicate = train_data.drop_duplicates()
    test_data_no_duplicate = test_data.drop_duplicates()

    x_train, y_train, x_test, y_test = preprocess_data(vectorizer, train_data_no_duplicate, test_data_no_duplicate)

    model = neural_network(x_train, y_train, x_test, y_test)

    agent()