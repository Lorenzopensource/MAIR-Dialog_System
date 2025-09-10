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
