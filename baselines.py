import pandas

# Load the pre-processed testing data
train_data = pandas.read_csv("train_data.csv")
test_data = pandas.read_csv("test_data.csv")

# Baseline 1: Most Frequent Class
most_frequent_class = train_data['dialog_act'].mode()[0]
print("Most frequent class in training data:", most_frequent_class)
baseline1_accuracy = (test_data['dialog_act'] == most_frequent_class).mean()
print("Baseline 1 (Most Frequent Class) Accuracy:", baseline1_accuracy)

# Baseline 2: Rule-Based (Word Presence)

keywords = {
    'ack': ['okay', 'ok', 'sure', 'alright'],
    'affirm': ['yes', 'yeah', 'yep', 'certainly'],
    'bye': ['bye', 'goodbye'],
    'confirm': ['confirm', 'correct', 'right'],
    'deny' : ['no', 'not'],
    'hello' : ['hello', 'hi', 'hey'],
    'inform' : ['information', 'informations','food','cuisine','price','european','eastern','japanese','type','cheap', 'expensive','chic', 'british','arab', 'thai','south', 'north', 'east', 'west', 'moroccan', 'italian', 'fancy','chinese','asian','oriental','indian','turkish','spanish', 'lebanese','sushi','romanian','welsh','nigerian','bbq','mean','fish'],  # the more food specific keywords, the better
    'negate' : ['no', 'not'],
    'null' : ['cough','laughter','silence','noise', 'background', 'inaudible', 'um' 'uh', 'hmm','sil', 'unintelligible'],
    'repeat' : ['repeat', 'say again', 'once more'],
    'reqalts' : ['others', 'alternatives', 'different','options','else','other','about'],
    'reqmore' : ['more', 'additional', 'extra','details'],
    'request' : ['can you', 'could you', 'would you', 'please','what', 'whats', 'where', 'when', 'which','adress','phone','postcode','may','area'],
    'restart' : ['restart', 'start over'],
    'thankyou' : ['thank you', 'thanks']
}

def rule_based_prediction(utterance):
    utterance = utterance.lower()
    for act, words in keywords.items():
        for word in words:
            if word in utterance:
                return act
    return 'not_covered'  # Default class if no keywords match

# Apply rule-based prediction on test data
test_data['predicted_act'] = test_data['utterance'].apply(rule_based_prediction)
baseline2_accuracy = (test_data['dialog_act'] == test_data['predicted_act']).mean()
print("Baseline 2 (Rule-Based) Accuracy:", baseline2_accuracy)

# Apply rule-based prediction on user input (asks for user input and predicts the dialog act, continuously until user types 'exit')
def user_input_prediction():
    while True:
        user_input = input("Enter an utterance (or type 'q' to quit): ")
        if user_input.lower() == 'q':
            break
        predicted_act = rule_based_prediction(user_input)
        print(f"Predicted Dialog Act: {predicted_act}")

user_input_prediction()
