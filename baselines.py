import pandas
from sklearn.metrics import classification_report

def most_frequent_class_baseline(train_data_path):
    train_data = pandas.read_csv(train_data_path)
    return train_data['dialog_act'].mode()[0]
    

keywords = {
    'request' : ['address','phone','postcode','what', 'whats', 'where','what\'s', 'may','please','area','post code','number', 'location'],
    'inform' : ['irish','french','mexican','austrian','australian','persian','greek','brazilian','mediterranean','bistro',
                'moderate','moderately','serves', 'any kind','german', 'dont know','either','doesnt matter','cuban','indonesian',
                'dont care','looking','information', 'informations','food','cuisine','price','european','eastern','japanese','type',
                'cheap', 'expensive','chic', 'british','arab', 'thai','south', 'north', 'east', 'west', 'moroccan', 'italian', 'fancy',
                'chinese','asian','oriental','indian','turkish','spanish', 'lebanese','sushi','romanian','welsh','nigerian','bbq','mean',
                'fish','korean','barbecue','fast food','catalan','pub','pizza','time','center','unusual','hungarian','african','american'],  # the more food specific keywords, the better
    'null' : ['cough','laughter','silence','noise', 'background', 'inaudible', 'sil', 'unintelligible','system','tv_noise'],
    'thankyou' : ['thank you', 'thanks',' thankyou', 'thank'],
    'reqmore' : ['more'],
    'reqalts' : ['how about','others', 'alternatives', 'different','options','else','other','another','what about','anything'],
    'bye': ['bye', 'goodbye'],
    'affirm': ['yes', 'yeah', 'yep', 'certainly','yea','correct','right'],
    'ack': ['okay', 'ok', 'sure', 'alright','kay'],
    'confirm': ['is it', 'is there','does it'],
    'negate' : ['no', 'not'],
    'deny' : ['wrong' 'don\'t', 'doesn\'t', 'isn\'t', 'wasn\'t', 'aren\'t', 'haven\'t','without','donot','dont'],
    'hello' : ['hello', 'hi', 'hey','ciao', 'ha', 'hay'],
    'repeat' : ['repeat', 'say again', 'once more', 'again'],
    'restart' : ['restart', 'start over']
}

# Note: changing keyword order can affect results due to early returns in the loop

def rule_based_prediction(utterance):
    utterance = utterance.lower()
    for act, words in keywords.items():
        for word in words:
            if word in utterance:
                return act
    return 'not_covered' 

if __name__ == "__main__":
    # Load the pre-processed testing data
    train_data = pandas.read_csv("utilities/train_data.csv")
    test_data = pandas.read_csv("utilities/test_data.csv")

    # Baseline 1: Most Frequent Class
    most_frequent_class = most_frequent_class_baseline(train_data_path="utilities/train_data.csv")
    print("Most frequent class in training data:", most_frequent_class)
    baseline1_accuracy = (test_data['dialog_act'] == most_frequent_class).mean()
    print("Baseline 1 (Most Frequent Class) Accuracy:", baseline1_accuracy)
    print("Classification report of baseline 1")
    predictions = [most_frequent_class] * len(test_data)
    print(classification_report(test_data["dialog_act"].tolist(), predictions, zero_division=1))

    # Baseline 2: Rule-Based (Word Presence)
    test_data['predicted_act'] = test_data['utterance'].apply(rule_based_prediction)
    baseline2_accuracy = (test_data['dialog_act'] == test_data['predicted_act']).mean()
    print("Baseline 2 (Rule-Based) Accuracy:", baseline2_accuracy)
    print("Classification report of baseline 2")
    predict_keyword = [rule_based_prediction(utterance) for utterance in test_data["utterance"]]
    print(classification_report(test_data["dialog_act"].to_list(), predict_keyword, zero_division=1))

    while True:
        user_input = input("Select the model to test (or 'q' to quit) -  Options: 'most_frequent' or 'rule_based': ")
        if user_input.lower() == 'q':
            break
        if user_input.lower() == 'most_frequent':
            user_input = input("Enter an utterance to classify (or 'q' to quit): ")
            if user_input.lower() == 'q':
                break
            print(f"Predicted Dialog Act: {most_frequent_class}")
            continue
        elif user_input.lower() == 'rule_based':
            user_input = input("Enter an utterance to classify (or 'q' to quit): ")
            if user_input.lower() == 'q':
                break
            predicted_act = rule_based_prediction(user_input)
            print(f"Predicted Dialog Act: {predicted_act}")
            continue
        else:
            print("Invalid option. Please choose 'most_frequent' or 'rule_based'.")
            
