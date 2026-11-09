Literary Time Classifier

Dataset
Period classes: 3
Book excerpts per class: 100
Total excerpts: 300

Source: biglam/blbooks-parquet (based off TheBritishLibrary/blbooks)

Class	                    Publication years
19th century (1st Third)	1800–1832
19th century (2nd Third)	1833–1866
19th century (3rd Third)	1867–1899


Task 1.
Get 100 English book excerpts from each century-third, that have sufficient OCR scores, and no author can contribute to more
than one excerpt across all centuries.


Task 2.
Design LLM classifier to predict the century-third of a book excerpt with Pydantic model structued output,
confidence score and explanation. Results are scored by accuracy.


Task 3.
Added accuracy, f1-score, recall, precision and confusion matrix to scoring.

Task 4. Add TF-IDF + Logistic Regression and TF-IDF (character n-grams) + Linear SVM

TO-DO:

1. Write README, and record results in there.

2. Implement
TypeScript Frontend, where user has option to put text and
each model responds with their prediction, confidence (except for svm), 
and in llm case, reasoning.

3. Update README 