import pandas as pd
from scipy.stats import ttest_ind

data = pd.read_csv("CleanedSurveyResults.csv")

formal = data[data["Formality"] == "Formal"]
informal = data[data["Formality"] == "Informal"]

for col in ["Interaction Quality", "Reliableness", "Overall Quality"]:
    stat, p = ttest_ind(formal[col], informal[col], equal_var=False)
    print(f"T-test for {col}: p-value = {p:.4f}")

formal_combined = formal[["Interaction Quality", "Reliableness", "Overall Quality"]].mean(axis=1)
informal_combined = informal[["Interaction Quality", "Reliableness", "Overall Quality"]].mean(axis=1)

stat, p_combined = ttest_ind(formal_combined, informal_combined, equal_var=False)
print(f"\nCombined Quality Score T-test: p-value = {p_combined:.4f}")
