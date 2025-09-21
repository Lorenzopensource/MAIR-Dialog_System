# MAIR-Dialog_System
This is the repository for the project assignment of **Methods in AI research**.

University of Utrecht, Academic Year 2025-2026.


## Authors

Daan van Dijk,	Lorenzo Pasqualotto, Harshavardhani Suresh Kumar and Ecenaz Rizeli

## Project Overview

This project is about the development of a restaurant recommendations dialog system using various methods from AI, such as domain modeling, text classification using machine learning and user experience testing. 

## Setting up the environment

1. **Install python 3**

2. **Create a python environment**:

- In your project folder run:
 ```bash
    python3 -m venv venv
```
3. **Activate the virtual environment**

- On macOS/Linux run:
 ```bash
    source venv/bin/activate
 ```
- On Windows (PowerShell):
 ```bash
    venv\Scripts\activate
 ```

4. **Install the libraries**:
 ```bash
   pip install -r requirements.txt
 ```
5. **Run the dialog system**:
 ```bash
   python dialog.py
 ```
6. **An example output of the dialog system**:
    To test the code, you can enter the following samples found at the bottom of the dialog.py. The following is an example of the output.
```plaintext
   System: Hello! welcome to restaurant search engine how can I help you?
    User:  Find a Cuban restaurant in the center
   System: In what area are you looking for a restaurant?
    User:  I want a restaurant in the west
   System: In what price range are you looking?
    User:  I want a moderately priced restaurant
    hierrrrrr
   System: So you are looking for a restaurant in west with thai food in the price range moderate right?
    User:  Yes
   System: We found this restaurant for you saint johns chop house
    you want another suggestion?
    User:  No
```


## Text Classification

The first thing a Dialogue System needs to do is to understand the user input.
In that sense we want to develope a mechanism that recognize and labels user intentions.

### Baselines ✅
After loading the dataset we constructed the required baselines.
You can check our performances and try a rule-based recognition of your intent running:
 ```bash
   python dataset_pre-processing.py
   python baselines.py
 ```

### Machine Learning Approach ✅
After generating the baseline models we started developping the machine learning models. We integrated three different classifiers: Logistic regression, support vectors machines and a neural network. 
 ```bash
   python machine_learning.py
 ```

### Evaluation ✅
Finally, we evaluated the models, including the baseline and the machine learning models. We did this using the classification report with the F1 scores, precision, and recall. The code is implemented in the existing python files baselines and machine learning.

To get all the details see the [final report](OVERVIEW.md).

### State transition diagram ✅
We decided to implement the user utterances implicitly in the state transition diagram. This ensures that the diagram remains clear without explicitly repeating the user utterance each time. The first 50 dialogues of the all dialogs were analyzed to find dialogue states and transitions. The examination showed that the three most common transitions were request area, request foodtype, and request price range. In some dialogues, the system asked for confirmation. This means that the choices already made by the user, such as area, food type, and price range, are repeated until they are confirmed by the user. Then the system provided a restaurant suggestion. Subsequently, the user could request information about the restaurant, such as the phone number, postal code, or address. Finally, the conversation ends if the user wants an alternative to the recommended restaurant.

### Dialog management ✅
 ```bash
  python dialog.py
 ```

