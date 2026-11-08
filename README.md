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


TO-DO:
1. Add accuracy, f1-score, recall, precision, confusion matrix to a function that can pass to llm classifier results
and other model results later


2. Implement

Model	Approach
Model 1	TF-IDF + Logistic Regression
Model 2	TF-IDF (character n-grams) + Linear SVM

3. Implement
TypeScript Frontend, where user has option to put text and
each model responds with their prediction, confidence (except for svm), 
and in llm case, reasoning.