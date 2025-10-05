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

5. **Run the dataset_pre-processing**:
```bash
   python dataset_pre-processing.py
 ```
7. **Run the baselines**:
```bash
   python dataset_pre-processing.py
 ```
9. **Run the machine_learning models**:
```bash
   python machine_learning.py
 ```
10. **Run the create_new_properties csv**:
```bash
   python create_new_properties.py
 ```
11. **Run the dialog.py**:
 ```bash
   python dialog.py
 ```

### Path Assumption

To avoid path dependencies problems, it is assumed that the function are runned inside the dedicated folder.
For example, to execute the final version of the dialog system, assuming you are currently on the root folder of the repo, run on your terminal:

 ```bash
   cd Part1
   python dialog.py
 ```


## Text Classification

The first thing a Dialogue System needs to do is to understand the user input.
In that sense we want to develope a mechanism that recognize and labels user intentions.

### Part 1A Baselines, Machine Learning and Evaluation
### Baselines 
After loading the dataset we constructed the required baselines.
You can check our performances and try a rule-based recognition of your intent running:
 ```bash
   python dataset_pre-processing.py
   python baselines.py
 ```
### Machine Learning Approach 
After generating the baseline models we started developping the machine learning models. We integrated three different classifiers: Logistic regression, support vectors machines and a neural network. 
 ```bash
   python machine_learning.py
 ```

### Evaluation 
Finally, we evaluated the models, including the baseline and the machine learning models. We did this using the classification report with the F1 scores, precision, and recall. The code is implemented in the existing python files baselines and machine learning. The baseline.py will print the classification reports immediately. However to print the classification reports for each  machine learning model you have to type evaluation in the output. 

### Part 1B and 1C Dialog Management

### State transition diagram 
We decided to implement the user utterances implicitly in the state transition diagram. This ensures that the diagram remains clear without explicitly repeating the user utterance each time. The first 50 dialogues of the all dialogs were analyzed to find dialogue states and transitions. The examination showed that the three most common transitions were request area, request foodtype, and request price range. In some dialogues, the system asked for confirmation. This means that the choices already made by the user, such as area, food type, and price range, are repeated until they are confirmed by the user. Then the system provided a restaurant suggestion. Subsequently, the user could request information about the restaurant, such as the phone number, postal code, or address. Finally, the conversation ends if the user wants an alternative to the recommended restaurant.


## Dialog management 
For part 1b, a working dialogue system interface has been implemented that prints system utterances on the screen and processes user input utterances using prediction classifiers. Furthermore, an algorithm has been implemented that identifies user preference statements in sentences by using keyword pattern matching and Levenshtein edit distance. A lookup function has also been set up that retrieves suitable restaurants from the database that match the preferences as extracted in the implemented algorithm.

For Part 1c, a new csv file called the restaurant_info_new_properties.csv file was created for the new properties: foodquality, crowdedness and lengthofstay by running the following file:
  ```bash
  python create_new_properties.py
 ```
Further, implication rules were implemented see report for clarification. We also handled the contradictions. Finally, five configurability features were implemented.

The following configurability features were implemented: 
- Output in CAP or not
- Delay in response or not
- Allow restarts or not
- Modify all the modifications 
- Modify the preferences

 ```bash
  python dialog.py
 ```

 ### An example output of the dialog system:
```plaintext
  ================================================================
  CONFIGURE YOUR PREFERENCES
  ================================================================
  Do you want to allow restarts?yes
  Do you want a small delay in the response?yes
  Do you want the output in CAP?yes

  ANSWER 'C' IF YOU WANT TO CHANGE THE MODIFICATIONS, 'R' IF YOU WANT TO RESTART AND 'M' IF YOU WANT TO MODIFY YOUR PREFERENCES

  ================================================================
  DIALOG
  ================================================================
   HELLO! WELCOME TO RESTAURANT SEARCH ENGINE HOW CAN I HELP YOU?
   User: I am looking for an indian restaurant
   System: IN WHAT AREA ARE YOU LOOKING FOR A RESTAURANT?
   User: north
   System: IN WHAT PRICE RANGE ARE YOU LOOKING?
   User: moderate
   System: SO YOU ARE LOOKING FOR A RESTAURANT IN NORTH WITH INDIAN FOOD IN THE PRICE RANGE MODERATE RIGHT?
   User: yes

   WE FOUND THESE RESTAURANT FOR YOU:
   ================================================================
    MEGHNA, THE NIRALA, TANDOORI PALACE, GRAFFITI
   ================================================================

   DO YOU HAVE ANY ADDITIONAL REQUIREMENT? 
   TYPE 1 FOR THE TOURISTIC RESTAURANTS
   TYPE 2 FOR THE ROMANTIC ONES
   TYPE 3 IF YOU WOULD LIKE THEM TO BE FOR CHILDREN
   TYPE 4 IF YOU WANT THOSE WHO PROVIDE ASSIGNED SEATS

  User: 2
 
  System: WE FOUND THIS RESTAURANT FOR YOU
  ================================================================
   TANDOORI PALACE
  ================================================================

  WOULD YOU LIKE ANY INFORMATION ABOUT THE RESTAURANT? 
  - TYPE 1 FOR PHONE NUMBER
  - TYPE 2 FOR ADDRESS
  - TYPE 3 FOR POSTCODE
  - TYPE 4 FOR ALL OF THEM
  - TYPE ANY OTHER CHARACTER TO EXIT THE SYSTEM

  User: 1

  ================================================================
  THE PHONE NUMBER OF THE RESTAURANT IS: 01223 360966.
  ================================================================

  DO YOU WANT ANY OTHER INFORMATION? 
  - TYPE 2 FOR ADDRESS
  - TYPE 3 FOR POSTCODE
  - TYPE 4 FOR ALL OF THEM
  - TYPE ANY OTHER CHARACTER TO EXIT THE SYSTEM

  User: X
  THANK YOU FOR USING OUR SERVICES, GOODBYE!
  RESTARTING.....
```




