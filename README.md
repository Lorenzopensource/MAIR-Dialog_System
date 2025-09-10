# MAIR-Dialog_System
This is the repository for the project assignment of **Methods in AI research**.

University of Utrecht, Academic Year 2025-2026.


--

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


## Text Classification

The first thing a Dialogue System needs to do is to understand the user input.
In that sense we want to develope a mechanism that recognize and labels user intentions.

### Baselines ✅
After loading the dataset we constructed the required baselines.
You can check our performances and try a rule-based recognition of your intent running:
 ```bash
   python baselines.py
 ```

### Machine Learning Approach ❌


### Dialog Management ❌

