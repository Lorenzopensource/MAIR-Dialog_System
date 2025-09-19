from sklearn.feature_extraction.text import CountVectorizer
import pandas
import joblib
import Levenshtein

def lookup(properties):
    df = pandas.read_csv("restaurant_info.csv")

    for key, value in properties.items():
        if key in df.columns:
            df = df[df[key].str.lower() == value]

    return df["restaurantname"].tolist()

def min_edit_distance(word, candidates):
    candidates = [c.lower() for c in candidates]
    distances = [(c, Levenshtein.distance(word, c)) for c in candidates]
    return min(distances, key=lambda x: x[1])[0] if candidates else None

def extract_property(user_input, candidates):
    words = user_input.lower().split()
    extracted = ""

    for w in words:
        if any(Levenshtein.distance(w, p.lower()) <= 2 for p in candidates):
            extracted = min_edit_distance(w, candidates)

    return extracted


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
        #info["context"]["area"] = user_input
        area = extract_property(user_input, valid_areas)
        if area:
            info["context"]["area"] = area
            return "check_area", f"You are looking for a restaurant in {user_input}?"
        else:
            return "ask_area", "We could not find that area, please try another one"

    if state == "ask_foodtype":
        #info["context"]["food type"] = user_input
        foodtype = extract_property(user_input, valid_foodtypes)
        if foodtype:
            info["context"]["foodtype"] = foodtype
            return "check_foodtype", f"You are looking for {user_input} food?"
        else:
            return "ask_foodtype", "We could not find that food type, please try another one"

    if state == "ask_price":
        #info["context"]["price range"] = user_input
        pricerange = extract_property(user_input, valid_priceranges)
        if pricerange:
            info["context"]["pricerange"] = pricerange
            return "check_price", f"You are looking for the price range {user_input}?"
        else:
            return "ask_price", "We could not find that price range, please try another one"

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

    restaurant_infos = pandas.read_csv("restaurant_info.csv")
    valid_areas = restaurant_infos["area"].dropna().unique().tolist()
    valid_foodtypes = restaurant_infos["food"].dropna().unique().tolist()
    valid_priceranges = restaurant_infos["pricerange"].dropna().unique().tolist()

    agent()

# --- TEST ---

    #input="I would like chinese food"
    #print(extract_property(input, valid_foodtypes))

    #input="I want a restaurant in the west"
    #print(extract_property(input, valid_areas))

    #input="I want a moderately priced restaurant "
    #print(extract_property(input, valid_priceranges))

# ---      ---