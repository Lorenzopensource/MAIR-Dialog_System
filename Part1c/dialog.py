from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import joblib
import Levenshtein
import time


def lookup(properties):
    df = pd.read_csv("restaurant_info.csv")

    for key, value in properties.items():
        if key in df.columns:
            df = df[df[key].str.lower() == value]

    return df["restaurantname"].tolist()


def min_edit_distance(word, candidates):
    candidates = [c.lower() for c in candidates]
    distances = [(c, Levenshtein.distance(word, c)) for c in candidates]
    return min(distances, key=lambda x: x[1])[0] if candidates else None


def extract_properties(user_input, candidates):
    words = user_input.lower().split()
    extracted = []

    for w in words:
        if any(Levenshtein.distance(w, p.lower()) <= 2 for p in candidates):
            extracted.append(min_edit_distance(w, candidates))

    return list(set(extracted))


# pre: restaurant is a string with the restaurant name
# post: returns a dictionary with the inferred properties of the restaurant
#       if a property cannot be inferred it will be set to None
#       if a property is inconsistent a function should handle the inconsistency (not implemented)
def inferred_properties(restaurant):
    df = pd.read_csv("restaurant_info_new_properties.csv")
    properties = df[df["restaurantname"] == restaurant.lower()]
    inferred = {"touristic": None, "romantic": None, "children": None, "assigned_seats": None}

    if properties.empty:
        return inferred

    # Rule 1
    if (properties["pricerange"].eq("cheap").any() and properties["foodquality"].eq("good").any()):
        inferred["touristic"] = True

    # Rule 2
    if properties["food"].eq("romanian").any():
        if inferred["touristic"] is True:
            print("Rule 2 is creating an inconsistency")  # handleInconsistency()
        else:
            inferred["touristic"] = False

    # Rule 3
    if properties["crowdness"].eq("busy").any():
        inferred["assigned_seats"] = True

    # Rule 4
    if properties["lengthofstay"].eq("long").any():
        inferred["children"] = False

    # Rule 5
    if properties["crowdness"].eq("busy").any():
        inferred["romantic"] = False

    # Rule 6
    if properties["lengthofstay"].eq("long").any():
        if inferred["romantic"] is False:
            print("Rule 6 is creating an inconsistency")  # handleInconsistency()
        else:
            inferred["romantic"] = True

    return inferred


def set_configs():
    restarts, delay, cap = False, False, False

    answer = input("Do you want to allow restarts?")

    vectorized = vectorizer.transform([answer])
    prediction = model.predict(vectorized)

    # Feature: allow restarts
    if prediction[0] == "affirm":
        restarts = True

    answer = input("Do you want a small delay in the response?")

    vectorized = vectorizer.transform([answer])
    prediction = model.predict(vectorized)

    # Feature: Return every output in caps lock
    if prediction[0] == "affirm":
        delay = True

    answer = input("Do you the output in CAP?")

    vectorized = vectorizer.transform([answer])
    prediction = model.predict(vectorized)

    # Feature: Return every output in caps lock
    if prediction[0] == "affirm":
        cap = True

    return restarts, delay, cap


def modify_preferences(info, cap, delay):
    if delay:
        time.sleep(1)

    if cap:
        user_input = input("Which preference do you want to change (area / food type / price range)?".upper())
    else:
        user_input = input("Which preference do you want to change (area / food type / price range)?")

    if user_input in ["food", "food type"]:
        user_input = "food type"
    elif user_input in ["price", "price range"]:
        user_input = "price range"
    elif user_input == "area":
        user_input = "area"
    else:
        log("Invalid choice. Please select 'area', 'food type', or 'price range'.", cap, delay)
        return info

    mapping = {
        "area": ("area", valid_areas),
        "food type": ("food", valid_foodtypes),
        "price range": ("pricerange", valid_priceranges),
    }

    key, candidates = mapping[user_input]

    if delay:
        time.sleep(1)
    if cap:
        new_value = input(f"Please enter the new {user_input}:".upper())
    else:
        new_value = input(f"Please enter the new {user_input}:\n")

    match = min_edit_distance(new_value, candidates)

    if not match:
        print(f"Sorry, we could not recognize that {user_input}. Keeping the old value.")
        return info

    info["context"][key] = match
    print(f"{user_input} updated to: {match}")

    return info


def log(message, cap, delay):
    if delay:
        time.sleep(1)
    if cap:
        print(message.upper())
    else:
        print(message)



def agent():
    info = {
        "context": {"area": "", "food": "", "pricerange": ""},
        "restaurants": [],
    }

    state = "start"
    user_input = ""

    # Ask the modifications of the agent
    print("Please first set the right modifications of the restaurant recommendation engine")

    restarts, delay, cap = set_configs()

    log("Answer 'c' if you want to change the modifications, 'r' if you want to restart and 'm' if you want to modify your preferences",cap, delay)

    # Feature: add a small delay to the response
    while True:
        oldstate = state

        state, message = state_transaction_function(state, user_input, info, cap, delay)
        if state == "end":
            log(message, cap, delay)

            if restarts:
                log("Restarting.....", cap, delay)

            break

        if delay:
            time.sleep(1)
        if cap:
            message = message.upper()

        user_input = input(f"{message}\n")

        if user_input == "c":
            restarts, delay, cap = set_configs()
            state = oldstate
            continue

        if user_input == "m":
            info = modify_preferences(info, cap, delay)
            user_input = ""
            state = "introduction"
            continue

        if user_input == "r":
            if not restarts:
                log("Restarting not allowed quiting....", cap, delay)
                break
            else:
                log("Restarting...", cap, delay)
                agent()


def state_transaction_function(state, user_input, info, cap, delay):
    if state == "start":
        return "introduction", "Hello! welcome to restaurant search engine how can I help you?"

    if state == "introduction":
        vectorized = vectorizer.transform([user_input])
        prediction = model.predict(vectorized)

        if prediction[0] == "inform":
            areas = extract_properties(user_input, valid_areas)
            foods = extract_properties(user_input, valid_foodtypes)
            prices = extract_properties(user_input, valid_priceranges)

            if areas:
                info["context"]["area"] = areas[0]
            if foods:
                info["context"]["food"] = foods[0]
            if prices:
                info["context"]["pricerange"] = prices[0]

            if not info["context"]["area"]:
                return "ask_area", "In what area are you looking for a restaurant?"
            if not info["context"]["food"]:
                return "ask_foodtype", "For what food type are you looking?"
            if not info["context"]["pricerange"]:
                return "ask_price", "In what price range are you looking?"
            else:
                return "confirmation", f"So you are looking for a restaurant in {info["context"]["area"]} with {info["context"]["food type"]} food in the price range {info["context"]["price range"]} right?"
        else:
            return "introduction", "Sorry we can't do that, try again please"

    if state == "ask_area":
        areas = extract_properties(user_input, valid_areas)
        if areas:
            info["context"]["area"] = areas[0]

            if not info["context"]["food"]:
                return "ask_foodtype", "For what food type are you looking?"
            elif not info["context"]["pricerange"]:
                return "ask_price", "In what price range are you looking?"
            else:
                return "confirmation", f"So you are looking for a restaurant in {info["context"]["area"]} with {info["context"]["food"]} food in the price range {info["context"]["pricerange"]} right?"
        else:
            return "ask_area", "We could not find that area, please try another one"

    if state == "ask_foodtype":
        foods = extract_properties(user_input, valid_foodtypes)
        if foods:
            info["context"]["food"] = foods[0]

            if not info["context"]["area"]:
                return "ask_area", "In what area are you looking for a restaurant?"
            elif not info["context"]["pricerange"]:
                return "ask_price", "In what price range are you looking?"
            else:
                return "confirmation", f"So you are looking for a restaurant in {info["context"]["area"]} with {info["context"]["food"]} food in the price range {info["context"]["pricerange"]} right?"
        else:
            return "ask_foodtype", "We could not find that food type, please try another one"

    if state == "ask_price":
        prices = extract_properties(user_input, valid_priceranges)
        if prices:
            info["context"]["pricerange"] = prices[0]
            if not info["context"]["area"]:
                return "ask_area", "In what area are you looking for a restaurant?"
            elif not info["context"]["food"]:
                return "ask_foodtype", "For what food type are you looking?"
            else:
                return "confirmation", f"So you are looking for a restaurant in {info["context"]["area"]} with {info["context"]["food"]} food in the price range {info["context"]["pricerange"]} right?"
        else:
            return "ask_price", "We could not find that price range, please try another one"

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
            return "confirmation", f"So you are looking for a restaurant in {info["context"]["area"]} with {info["context"]["food"]} food in the price range {info["context"]["pricerange"]} right?"
        else:
            return "ask_price", "In what price range are you looking?"

    if state == "confirmation":
        vectorized = vectorizer.transform([user_input])
        prediction = model.predict(vectorized)

        if prediction[0] == "affirm":
            info["restaurants"] = lookup(info["context"])

            if info["restaurants"]:
                n = 5  # number of restaurants to show
                n_restaurants = info["restaurants"][:n] if len(info["restaurants"]) > n else info["restaurants"]
                log(f"We found these restaurant for you: \n {', '.join(n_restaurants)}", cap , delay)
                return "end", "Enjoy your meal!"

            else:
                return "end", "We could not find a restaurant for you restarting if possible!"

        else:
            log("Sorry for the misunderstanding... What did I got wrong? Area, food type or price range?", cap, delay)
            clarification = min_edit_distance(input().lower(), ["area", "food", "price"])
            if clarification == "area":
                return "ask_area", "Got it! In what area are you looking for a restaurant?"
            elif clarification == "food":
                return "ask_foodtype", "Got it! For what food type are you looking?"
            elif clarification == "price":
                return "ask_price", "Sorry for the misunderstanding... In what price range are you looking?"
            else:
                return "confirmation", f"So you are looking for a restaurant in {info["context"]["area"]} with {info["context"]["food"]} food in the price range {info["context"]["pricerange"]} right?"


if __name__ == "__main__":
    model = joblib.load('Utterance_Classifier_NN.pkl')
    vectorizer = joblib.load('Vectorizer_NN.pkl')

    restaurant_infos = pd.read_csv("restaurant_info.csv")

    valid_areas = restaurant_infos["area"].dropna().unique().tolist()
    valid_foodtypes = restaurant_infos["food"].dropna().unique().tolist()
    valid_priceranges = restaurant_infos["pricerange"].dropna().unique().tolist()

    agent()
