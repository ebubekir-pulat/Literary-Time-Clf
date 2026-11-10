# Literary Time Classifier

Classifies English book excerpts into three periods of the 19th century:

- **1800–1832**
- **1833–1866**
- **1867–1899**

## Current status

- Dataset preparation is implemented using `biglam/blbooks-parquet`
  (this dataset is based off TheBritishLibrary/blbooks).
- The preparation script targets 100 excerpts per period, filters for English-language text and OCR quality, and avoids repeated authors/publication names/record IDs.
- Train/test loading and an 80/20 split are implemented.
- Three classifiers are implemented:
  - **Qwen 2.5 3B (Ollama):** structured prediction, confidence score, and explanation.
  - **TF-IDF + Logistic Regression**
  - **Character TF-IDF (1–3 grams) + Linear SVM**
- Evaluation reports accuracy, precision, recall, F1-score, and a confusion matrix.

## Project structure

```text
results/
    LLM_Cfs_Mtx.png
    LLM_clf_report.png
backend/
  requirements.txt
  data/
  scripts/
    prepare_dataset.py
    load_dataset.py
    LLM_Classifier.py
    ML_Classifier.py
    score.py
```

## Setup and run

Install Python dependencies:

```bash
pip install -r backend/requirements.txt
```

Prepare the dataset:

```bash
cd backend/scripts
python prepare_dataset.py
```

Then run a classifier from `backend/scripts`:

```bash
python ML_Classifier.py
# or
python LLM_Classifier.py
```

The LLM classifier requires Ollama and the `qwen2.5:3b` model to be available locally.

## Model Results

qwen2.5:3b Results

![Alt Text](results/LLM_clf_report.png)

![Alt Text](results/LLM_Cfs_Mtx.png)

## Outstanding work / limitations

- The dataset JSON files are not included in the supplied project archive; generate them before training or evaluation.
- TO-DO: No sklearn model evaluation results are currently recorded.
- TO-DO: The TypeScript frontend is not implemented.
- Saved model files are written to the current working directory.
- The scoring script uses Matplotlib's `TkAgg` backend, which may require a local GUI environment.



























 






TO-DO:

2. Implement
TypeScript Frontend, where user has option to put text and
each model responds with their prediction, confidence (except for svm), 
and in llm case, reasoning.

3. Update README 