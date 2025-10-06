import pandas 
import sklearn
from sklearn.model_selection import train_test_split

dataset_path = 'utilities/dialog_acts.dat' 

# Loading the dataset

with open(dataset_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

data = []
for line in lines:
    line = line.strip()
    parts = line.split(maxsplit=1)  
    dialog_act, utterance = parts
    data.append([dialog_act, utterance])

dataset = pandas.DataFrame(data, columns=["dialog_act", "utterance"])

# Lowercasing the text

dataset["dialog_act"] = dataset["dialog_act"].str.lower()
dataset["utterance"] = dataset["utterance"].str.lower()

# Splitting the dataset into training and testing sets (85% train, 15% test)
train_data, test_data = train_test_split(dataset,  test_size=0.15, random_state=142)

# Saving dataset
train_data.to_csv("utilities/train_data.csv", index=False)
test_data.to_csv("utilities/test_data.csv", index=False)