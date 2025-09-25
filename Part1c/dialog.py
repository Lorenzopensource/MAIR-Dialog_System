from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import joblib
import Levenshtein


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
    if (properties["pricerange"].eq("cheap").any()  and properties["foodquality"].eq("good").any()):
        inferred["touristic"] = True

    # Rule 2
    if properties["food"].eq("romanian").any():
        if inferred["touristic"] is True:
            print("Rule 2 is creating an inconsistency") # handleInconsistency()
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
            print("Rule 6 is creating an inconsistency") # handleInconsistency()
        else:
            inferred["romantic"] = True

    return inferred



def agent():
    info = {
        "context": {"area": "", "food": "", "pricerange": "" , "addReq": ""},
        "restaurants": [],
    }

    state = "start"
    user_input = ""

    while True:
        state, message = state_transaction_function(state, user_input, info)
        if state == "end":
            print(message)
            break

        user_input = input(f"{message}\n").lower()


def state_transaction_function(state, user_input, info):

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
                n = 5 # number of restaurants to show
                n_restaurants = info["restaurants"][:n] if len(info["restaurants"]) > n else info["restaurants"]
                print(f"We found these restaurant for you: \n {', '.join(n_restaurants)}")
                if info["context"]["addReq"] == " " :
                    return "ask_add" , "Do you have any additional requirement about the listed restaurants ? /n  Select 1 for touristic restaurants places /n Select 2 for Romantic restaurant places /n Select 3 for Making special resevation for Children /n Select 4 to specify the number of seat to be reserved: /n"
                else:  return "end", "Enjoy your meal!"

            else:
                return 'ask_area', "We could not find a restaurant for you, restarting......"

        else:
            print("Sorry for the misunderstanding... What did I got wrong? Area, food type or price range?")
            clarification = min_edit_distance(input().lower(), ["area", "food", "price"])
            if clarification == "area":
                return "ask_area", "Got it! In what area are you looking for a restaurant?"
            elif clarification == "food":
                return "ask_foodtype", "Got it! For what food type are you looking?"
            elif clarification == "price":
                return "ask_price", "Sorry for the misunderstanding... In what price range are you looking?"
            else:
                return "confirmation", f"So you are looking for a restaurant in {info["context"]["area"]} with {info["context"]["food"]} food in the price range {info["context"]["pricerange"]} right?"
            
    if state == "ask_add":
        if user_input == 1:
            info["context"]["addReq"] = "touristic"
        if user_input == 2:
            info["context"]["addReq"] = "romantic"
        if user_input == 3:
            info["context"]["addReq"] = "children"
        if user_input == 4 :
            info["context"]["addReq"] = "assigned_seats"
        else:
            return "ask_add","Please enter a valid option"
        
        filtered = filter_req(info["restaurants"],info["context"]["addReq"])
        if not filtered:  return 'ask_add', 'We could not find any restaurant with the specified requirement, please insert a new one.'
        else: 
            print(f"We found these restaurant for you: \n {', '.join(filtered)}")
            return "end", "Enjoy your meal!"



def filter_req(restaurants,addReq):
    filtered = []
    for restaurant in restaurants:
        if inferred_properties(restaurant)[addReq]:
            filtered.append(restaurant)
    return filtered

         



if __name__ == "__main__":

    model = joblib.load('Utterance_Classifier_NN.pkl')
    vectorizer = joblib.load('Vectorizer_NN.pkl')

    restaurant_infos = pd.read_csv("restaurant_info.csv")

    valid_areas = restaurant_infos["area"].dropna().unique().tolist()
    valid_foodtypes = restaurant_infos["food"].dropna().unique().tolist()
    valid_priceranges = restaurant_infos["pricerange"].dropna().unique().tolist()

    agent()

# --- TESTS ---
#
#    input="Find a Cuban restaurant in the center"
#    print(extract_properties(input, valid_foodtypes))
#
#    input="I want a restaurant in the west"
#    print(extract_property(input, valid_areas))
#
#    input="I want a moderately priced restaurant "
#    print(extract_property(input, valid_priceranges))

#    print(inferred_properties("the gardenia"))

# ---      ---