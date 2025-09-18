## Quantitative Evaluation

We chose to evaluate the models using a classification report that shows the precision,
recall, F1 scores, and accuracy for the classes of each model. It was already clear that the
**inform** class was the dominant class in each model. This meant that we were dealing with an
unbalanced dataset. Because of this we looked at the accuracy and weighted average of the
F1 scores to assess how the models perform relative to each other.

### Performance of the models

- **Baseline 1** shows a poor performance as it gives the lowest scores in the classification
  report.
- **Baseline 2** performs better than Baseline 1, as shown in its accuracy and F1-score.
- The **machine learning models** perform roughly the same with the non-deduplicated data,
  when we look at the accuracy and weighted average of F1 scores.
- The **MLP classifier** performs slightly better with a very small difference.  
  This does not change when we remove duplicates from the dataset; we only
  observe that the accuracy and weighted average of F1 scores decrease slightly, with MLP
  still performing best.

#### Model Comparison

| Model                       | Accuracy | F1-score |
|------------------------------|----------|----------|
| Baseline 1                  | 0.39     | 0.22     |
| Baseline 2                  | 0.80     | 0.80     |
| Logistic Regression (non-deduplicated) | 0.987    | 0.99     |
| Logistic Regression (deduplicated)     | 0.95     | 0.94     |
| SVM (non-deduplicated)      | 0.986    | 0.99     |
| SVM (deduplicated)          | 0.95     | 0.94     |
| MLP (non-deduplicated)      | 0.99     | 0.99     |
| MLP (deduplicated)          | 0.97     | 0.97     |

---

### Difficult Cases

#### 1. Negations
The first difficult instance focuses on the presence of a negation which can be classified as
**negate** by some systems while others classify it as **inform**.

| Test instance                                                                 | Logistic Regression | SVM   | MLP   |
|-------------------------------------------------------------------------------|---------------------|-------|-------|
| *No, I do not want turkish food, I want Italian food*                         | Negate              | Inform| Negate|
| *No I do not want a cheap restaurant, I want an expensive restaurant*         | Negate              | Inform| Negate|

#### 2. Spelling Errors
Minor spelling errors cause incorrect classification as shown below, where all three models
label these words as **inform** when they belong under *hello*, *thankyou*, and *bye*.

| Test instance | Logistic Regression | SVM   | MLP   |
|---------------|---------------------|-------|-------|
| Thankyoi      | Inform              | Inform| Inform|
| by            | Inform              | Inform| Inform|
| hllo          | Inform              | Inform| Inform|

---

## Error Analysis

We analyzed the misclassifications of the models to understand which dialog acts and utterances are more difficult to classify.


### Top 3 Most Frequent Misclassified Dialog Acts

#### Logistic Regression
| Dialog Act | Count |
|------------|-------|
| inform     | 27    |
| request    | 12    |
| affirm     | 9     |

#### SVM
| Dialog Act | Count |
|------------|-------|
| inform     | 20    |
| request    | 15    |
| affirm     | 13    |

#### Neural Network (MLP)
| Dialog Act | Count |
|------------|-------|
| inform     | 19    |
| negate     | 4     |
| thankyou   | 4     |

#### Rule-Based (Baseline 2)
| Dialog Act | Count |
|------------|-------|
| inform     | 546   |
| request    | 243   |
| reqalts    | 137   |

---

### Utterances Misclassified by All Systems

**Number of sentences misclassified by all systems:** 18  

| Utterance                           | True Label | Logistic Regression | SVM   | Neural Network | Rule-Based |
|------------------------------------|------------|-------------------|-------|----------------|------------|
| nor                                | ack        | reqalts           | reqalts | reqalts        | confirm    |
| thank you good bye                  | reqalts    | inform            | inform | inform         | confirm    |
| can i have the address and the postcode | inform  | request           | request | request       | confirm    |
| thank you good bye                  | inform     | confirm           | reqalts | confirm       | confirm    |
| phone number                        | negate     | inform            | inform | request        | confirm    |
| cheap                               | request    | inform            | inform | inform         | confirm    |
| indian                              | reqalts    | request           | request | request       | confirm    |
| moderately                          | deny       | inform            | inform | inform         | confirm    |
| and what kind of food               | ack        | inform            | inform | inform         | confirm    |
| address                             | inform     | request           | request | request       | confirm    |
| thank you good bye                  | bye        | thankyou          | thankyou | thankyou     | confirm    |
| the address                         | bye        | thankyou          | thankyou | thankyou     | confirm    |
| what is the address                 | inform     | reqalts           | reqalts | reqalts       | confirm    |
| moderately priced restaurant        | confirm    | inform            | inform | inform         | confirm    |
| greek                               | deny       | inform            | inform | inform         | confirm    |
| gastropub                           | confirm    | inform            | reqalts | reqalts       | confirm    |
| yes                                 | inform     | reqalts           | reqalts | reqalts       | confirm    |

---

### Observations

- The **inform** dialog act is the most frequently misclassified across all models, indicating it is particularly challenging, likely due to its high frequency and semantic variability.
- **Request** and **reqalts** are also common sources of error, especially in SVM and Logistic Regression.
- The **rule-based system** (Baseline 2) struggles significantly more than the machine learning models, particularly with **inform**, **request**, and **reqalts**.
- Some utterances, such as `"thank you good bye"`, `"phone number"`, and `"moderately priced restaurant"`, are consistently misclassified by all systems, highlighting intrinsic ambiguity or lack of context. The Baseline 2 is an exception because we had the possibility to check in the training set how those instances where commonly labeled and choose the discriminating word.
- Errors often arise due to:
  - Short or incomplete utterances.
  - Ambiguous phrasing or multiple intents.
  - Negations and polite forms that can resemble affirmations.
  - Spelling or lexical variations (as noted in the previous section).


