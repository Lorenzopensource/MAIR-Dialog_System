from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import joblib
import Levenshtein
import time

# pre: properties is a dictionary with keys as column names and values as the desired values for those columns
# post: returns a list of restaurant names that match the given properties
def lookup(properties):
    df = pd.read_csv("restaurant_info.csv")

    for key, value in properties.items():
        if value == "any":
            continue
        elif key in df.columns:
            df = df[df[key].str.lower() == value]

    return df["restaurantname"].tolist()


# pre: word is a string, candidates is a list of strings
# post: returns the candidate with the minimum edit distance to the word
def min_edit_distance(word, candidates):
    candidates = [c.lower() for c in candidates]
    distances = [(c, Levenshtein.distance(word, c)) for c in candidates]
    return min(distances, key=lambda x: x[1])[0] if candidates else None

# pre: user_input is a string, candidates is a list of strings
# post: returns a list of candidates that are present in the user_input with an edit distance of at most 1
def extract_properties(user_input, candidates):
    words = user_input.lower().split()
    extracted = []

    for w in words:
        if any(Levenshtein.distance(w, p.lower()) <= 1 for p in candidates):
            extracted.append(min_edit_distance(w, candidates))

    return list(set(extracted))

# pre: restaurant is a name of a restaurant, addReq is one of the additional requirements
# post: returns "Yes" if the restaurant satisfies the additional requirement, "No" if it does not, "Unknown" if it cannot be inferred
#       "Inconsistency" if the properties of the restaurant lead to an inconsistency
def has_inferred_property(restaurant, add_req):
    df = pd.read_csv("restaurant_info_new_properties.csv")
    properties = df[df["restaurantname"] == restaurant.lower()]
    if add_req == "touristic":
        # Rule 1: "If a restaurant is cheap and has good food quality then it is touristic"
        if properties["pricerange"].eq("cheap").any() and properties["foodquality"].eq("good").any():
            # Check for internal inconsistencies
            # Rule 2: "If a restaurant serves Romanian food then it is not touristic"
            if properties["food"].eq("romanian").any():
                return "Inconsistency"
            return "Yes"
        elif properties["food"].eq("romanian").any():
            return "No"
        else:
            return "Unknown"
    if add_req == "assigned_seats":
    # Rule 3: "If a restaurant is busy then it provides assigned seats"
        if properties["crowdness"].eq("busy").any():
            return "Yes"
        else:
            return "Unknown"
    if add_req == "children":
        # Rule 4: "If a restaurant has a long length of stay then it is not suitable for children",
        if properties["lengthofstay"].eq("long").any():
            return "No"
        else:
            return "Unknown"
    if add_req == "romantic":
        # Rule 5: "If a restaurant is busy then it is not romantic",
        if properties["crowdness"].eq("busy").any():
            # Check for internal inconsistencies
            # Rule 6: "If a restaurant has a long length of stay then it is romantic",
            if properties["lengthofstay"].eq("long").any():
                return "Inconsistency"
            return "No"
        elif properties["lengthofstay"].eq("long").any():
            return "Yes"
        else:
            return "Unknown"
    else: return "Unknown"    

# pre: restaurants is a list of restaurant names, addReq is one of the additional requirements
# post: returns True and a restaurant name if there is at least one restaurant satisfying the additional requirement
#       returns False and a message if no restaurant satisfies the additional requirement or if there is an inconsistency
def filter_add_req(restaurants,addReq):
    checks = []
    for restaurant in restaurants:
        checks.append(has_inferred_property(restaurant,addReq))
    inconsistency = False
    for i in range(len(restaurants)):
        if checks[i] == "Yes":
            return True, restaurants[i]
        if checks[i] == "Inconsistency":
            inconsistency = True
    if inconsistency:
        return False,"The additional requirement cannot be determined for the selected restaurants, please specify another one"
    return False,"We could not find any restaurant with the specified requirement, please insert a new one."

# Ask the user for the configurations of the dialog system
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

    answer = input("Do you want the output in CAP?")

    vectorized = vectorizer.transform([answer])
    prediction = model.predict(vectorized)

    # Feature: Return every output in caps lock
    if prediction[0] == "affirm":
        cap = True

    return restarts, delay, cap

# Allow the user to modify their preferences
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

# Log messages with optional capitalization and delay
def log(message, cap, delay):
    if delay:
        time.sleep(1)
    if cap:
        print(message.upper())
    else:
        print(message)

# Main dialog function
def agent():
    info = {
        "context": {"area": "", "food": "", "pricerange": "", "addReq": ""},
        "restaurants": [],
    }

    state = "start"
    user_input = ""

    # Ask the modifications of the agent
    print("\n================================================================ \nCONFIGURE YOUR PREFERENCES \n================================================================ \n ")

    restarts, delay, cap = set_configs()

    log("\nAnswer 'c' if you want to change the modifications, 'r' if you want to restart and 'm' if you want to modify your preferences \n \n================================================================ \nDIALOG \n================================================================ \n",cap, delay)

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

        if user_input == "c" or user_input == "C":
            restarts, delay, cap = set_configs()
            user_input = ""
            state = "introduction"
            continue

        if user_input == "m" or user_input == "M":
            info = modify_preferences(info, cap, delay)
            user_input = ""
            state = "introduction"
            continue

        if user_input == "r" or user_input == "R":
            if not restarts:
                log("Restarting not allowed quiting....", cap, delay)
                break
            else:
                log("Restarting...", cap, delay)
                agent()

# State transition function
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
                return "confirmation", f"So you are looking for a restaurant in {info['context']['area']} with {info['context']['food']} food in the price range {info['context']['pricerange']} right?"
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
                return "confirmation", f"So you are looking for a restaurant in {info['context']['area']} with {info['context']['food']} food in the price range {info['context']['pricerange']} right?"
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
                return "confirmation", f"So you are looking for a restaurant in {info['context']['area']} with {info['context']['food']} food in the price range {info['context']['pricerange']} right?"
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
                return "confirmation", f"So you are looking for a restaurant in {info['context']['area']} with {info['context']['food']} food in the price range {info['context']['pricerange']} right?"
        else:
            return "ask_price", "We could not find that price range, please try another one"

    if state == "confirmation":
        vectorized = vectorizer.transform([user_input])
        prediction = model.predict(vectorized)

        if prediction[0] == "affirm":
            info["restaurants"] = lookup(info["context"])

            if info["restaurants"]:
                n = 5  # number of restaurants to show
                n_restaurants = info["restaurants"][:n] if len(info["restaurants"]) > n else info["restaurants"]
                log(f"\n We found these restaurants for you: \n================================================================  \n {', '.join(n_restaurants)} \n================================================================  \n", cap , delay)
                if info["context"]["addReq"] == "" :
                    return "ask_add" , "Do you have any additional requirement? \n Type 1 for the touristic restaurants \n Type 2 for the romantic ones\n Type 3 if you would like them to be for children  \n Type 4 if you want those who provide assigned seats  \n"
                else:
                    return "end", "Enjoy your meal!"

            else:
                info["context"] = {"area": "", "food": "", "pricerange": "", "addReq": ""}
                return "ask_area", "We could not find a restaurant for you! Let's start again! \nIn what area are you looking for a restaurant?"

        else:
            log("Sorry for the misunderstanding... What did I got wrong? Area, food type or price range?", cap, delay)
            clarification = min_edit_distance(input().lower(), ["area", "food", "price"])
            if clarification == "area":
                return "ask_area", "Got it! In what area are you looking for a restaurant?"
            elif clarification == "food":
                return "ask_foodtype", "Got it! For what food type are you looking?"
            elif clarification == "price":
                return "ask_price", "Got it! What price range do you prefer?"
            else:
                return "confirmation", f"So you are looking for a restaurant in {info['context']['area']} with {info['context']['food']} food in the price range {info['context']['pricerange']} right?"

    if state == "ask_add":
        if user_input == "1":
            info["context"]["addReq"] = "touristic"
        elif user_input == "2":
            info["context"]["addReq"] = "romantic"
        elif user_input == "3":
            info["context"]["addReq"] = "children"
        elif user_input == "4":
            info["context"]["addReq"] = "assigned_seats"
        else:
            return "ask_add", "Please enter a valid option"

        found, text = filter_add_req(info["restaurants"], info["context"]["addReq"])
        if not found:
            return 'ask_add', text
        else:
            log(f' \nWe found this restaurant for you: \n================================================================ \n  {text} \n================================================================ \n', cap , delay)
            return "provide_info", "Would you like any information about the restaurant? \n - Type 1 for phone number \n - Type 2 for address \n - Type 3 for postcode \n - Type 4 for all of them \n - Type any other character to exit the system \n"
        
    if state == "provide_info":
        df = pd.read_csv("restaurant_info.csv")
        restaurant_name = filter_add_req(info["restaurants"], info["context"]["addReq"])[1]

        restaurant_infos = df[df["restaurantname"] == restaurant_name.lower()]

        if user_input == "1":
            log(f"\n================================================================ \n  The phone number of the restaurant is: {restaurant_infos['phone'].values[0]}. \n================================================================ \n ", cap, delay)
            return "provide_info", "Do you want any other information? \n - Type 2 for address \n - Type 3 for postcode \n - Type 4 for all of them \n - Type any other character to exit the system \n"
        elif user_input == "2":
            log(f"\n================================================================ \n  The address of the restaurant is: {restaurant_infos['addr'].values[0]}. \n================================================================ \n ", cap, delay)
            return "provide_info",  "Do you want any other information? \n - Type 1 for phone number \n - Type 3 for postcode \n - Type 4 for all of them \n - Type any other character to exit the system \n"
        elif user_input == "3":
            log(f"\n================================================================ \n  The postcode of the restaurant is: {restaurant_infos['postcode'].values[0]}. \n================================================================ \n ", cap, delay)
            return "provide_info", "Do you want any other information? \n - Type 1 for phone number \n - Type 2 for address \n - Type 4 for all of them \n - Type any other character to exit the system \n"
        elif user_input == "4":
            log(f"\n================================================================ \nPhone number: {restaurant_infos['phone'].values[0]} \nAddress: {restaurant_infos['addr'].values[0]} \nPostcode: {restaurant_infos['postcode'].values[0]} \n================================================================ \n ", cap, delay )
            return "end","Thank you for using our services, goodbye!"
        else:
            return "end", "Thank you for using our services, goodbye!"

if __name__ == "__main__":
    model = joblib.load('Utterance_Classifier_NN.pkl')
    vectorizer = joblib.load('Vectorizer_NN.pkl')

    restaurant_infos = pd.read_csv("restaurant_info.csv")

    valid_areas = restaurant_infos["area"].dropna().unique().tolist()
    valid_foodtypes = restaurant_infos["food"].dropna().unique().tolist()
    valid_priceranges = restaurant_infos["pricerange"].dropna().unique().tolist()

    valid_areas.append("any")
    valid_foodtypes.append("any")
    valid_priceranges.append("any")

    agent()