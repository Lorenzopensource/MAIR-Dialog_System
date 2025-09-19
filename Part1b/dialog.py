from sklearn.feature_extraction.text import CountVectorizer
import pandas
import joblib


def lookup(properties):
    df = pandas.read_csv("restaurant_info.csv")

    for key, value in properties.items():
        if key in df.columns:
            df = df[df[key].str.lower() == value]

    return df["restaurantname"].tolist()


def agent():
    info = {
        "context": {"area": "", "food type": "", "price range": ""},
        "restaurants": [],
        "number": -1,
    }

    state = "start"
    user_input = ""

    while True:
        state, message = state_transaction_function(state, user_input, info)

        user_input = input(f"{message}\n").lower()


def state_transaction_function(state, user_input, info):

    print(info["restaurants"])

    if state == "start":
        return "introduction", "Hello! welcome to restaurant search engine how can I help you?"

    if state == "introduction":
        vectorized = vectorizer.transform([user_input])
        prediction = model.predict(vectorized)

        if prediction[0] == "inform":
            return "ask_area", "in what area you looking for a restaurant"

    if state == "ask_area":
        info["context"]["area"] = user_input
        return "check_area", f"You are looking for a restaurant in {user_input}?"

    if state == "ask_foodtype":
        info["context"]["food type"] = user_input
        return "check_foodtype", f"You are looking for {user_input} food?"

    if state == "ask_price":
        info["context"]["price range"] = user_input
        return "check_price", f"You are looking for the price range {user_input}?"

    if state == "check_area":
        vectorized = vectorizer.transform([user_input])
        prediction = model.predict(vectorized)

        if prediction[0] == "affirm":
            return "ask_foodtype", "for what food type are you looking?"
        else:
            return "ask_area", "in what area you looking for a restaurant"

    if state == "ask_another":
        vectorized = vectorizer.transform([user_input])
        prediction = model.predict(vectorized)

        if prediction[0] == "affirm":
            info["number"] += 1

            print(f"We found this restaurant for you {info["restaurants"][info["number"]]}")

            if len(info["restaurants"]) > info["number"] + 1:
                return "ask_another", "you want another suggestion?"
            else:
                return "start", "No more restaurants found"

        else:
            agent()

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
            return "confirmation", f"So you are looking for a restaurant in {info["context"]["area"]} with {info["context"]["food type"]} food in the price range {info["context"]["price range"]} right?"
        else:
            return "ask_price", "In what price range are you looking?"

    if state == "confirmation":
        vectorized = vectorizer.transform([user_input])
        prediction = model.predict(vectorized)

        if prediction[0] == "affirm":
            info["restaurants"] = lookup(info["context"])

            if info["restaurants"]:
                info["number"] += 1
                print(f"We found this restaurant for you {info["restaurants"][info['number']]}")

                if len(info["restaurants"]) > info["number"] + 1:
                    return "ask_another", "you want another suggestion?"
                else:
                    return "start", "No more restaurants found"
            else:
                print("We could not find a restaurant for you, restarting......")
                agent()

        else:
            return "ask_area", "in what area you looking for a restaurant"


if __name__ == "__main__":

    model = joblib.load('Utterance_Classifier_NN.pkl')
    vectorizer = joblib.load('Vectorizer_NN.pkl')

    agent()