# Email Spam Classification

An end-to-end NLP classification project that detects spam from raw email text. The repository covers exploratory analysis, hypothesis testing, leakage-safe preprocessing, TF-IDF feature engineering, feature selection, model comparison, hyperparameter tuning, threshold selection, independent final evaluation, error analysis, and API deployment.

## Project Highlights

- Built a reproducible text-classification workflow for **5,728 labeled emails**.
- Reserved a **locked final test set before EDA and model selection**.
- Kept normalized duplicate emails in the same data split to prevent leakage.
- Compared word and character representations, feature-selection methods, eight classical classifiers, and voting ensembles.
- Selected a precision-aware decision threshold on validation data only.
- Packaged the complete cleaner, TF-IDF transformation, classifier, and threshold as a reusable artifact.
- Added a production-style FastAPI service, Docker image, automated API tests, and Google Cloud Run configuration.

## Problem

The task is binary classification:

- `0` — legitimate email (`ham`)
- `1` — spam

False positives are especially costly because a legitimate message sent to the spam folder can be missed. For this reason, the final threshold was selected with a minimum validation precision of `0.99`, while maximizing recall among the eligible thresholds.

## Data

The dataset contains two columns:

| Column | Description |
|---|---|
| `text` | Raw email subject and body |
| `spam` | Binary target: `1` for spam, `0` for legitimate email |

The dataset has **5,728 rows**, with spam representing approximately **23.9%** of the observations. It is the public email dataset used in MIT's *The Analytics Edge* text-analytics assignment, assembled from Enron ham messages and several public spam corpora.

Dataset reference: [MIT OpenCourseWare — Text Analytics Assignment](https://ocw.mit.edu/courses/15-071-the-analytics-edge-spring-2017/pages/text-analytics/assignment-5/)

## Methodology

1. **Independent split** — normalized email groups were split before analysis, preventing near-identical messages from crossing between development and final test data.
2. **EDA and hypothesis testing** — analyzed length, spam vocabulary, punctuation, noisy spelling patterns, URLs, financial language, digits, and calls to action.
3. **Cleaning** — normalized technical patterns such as URLs, email addresses, HTML, money values, and numbers while preserving useful linguistic signal.
4. **Outlier experiments** — compared raw counts, scaling, `log1p`, Yeo–Johnson, clipping, and quantile binning. Removing extreme emails did not improve validation quality.
5. **Feature engineering** — compared CountVectorizer, word TF-IDF, character TF-IDF, and a combined word + character representation.
6. **Feature selection** — tested no additional selection, chi-square selection, and L1-based selection.
7. **Model comparison** — evaluated Logistic Regression, Linear SVC, SGD, Ridge, three Naive Bayes variants, Dummy baseline, and voting ensembles.
8. **Tuning** — used GridSearchCV and RandomizedSearchCV only on development folds.
9. **Threshold selection** — used the validation set to choose a high-precision spam threshold.
10. **Locked final test** — opened once after all modeling decisions were fixed.

## Final Results

The final model is a **Linear SVC** using combined word and character TF-IDF features.

| Metric | Locked final test |
|---|---:|
| Accuracy | **0.9915** |
| Precision | **0.9901** |
| Recall | **0.9756** |
| F1 | **0.9828** |
| ROC-AUC | **0.9996** |
| PR-AUC / Average Precision | **0.9988** |
| False positives | **2** |
| False negatives | **5** |

The locked test set contained **825 emails** and had zero overlap with normalized development texts.

## Error Analysis

The model struggled mainly with two opposite cases:

- very short spam messages with little context or recognizable spam vocabulary;
- long spam messages written like technical or professional correspondence.

The two false positives contained URLs and words such as `free` and `download`, making legitimate messages resemble promotional email. These cases show why threshold control and manual review remain useful even with strong aggregate metrics.

## Repository Structure

```text
.
├── Email_Spam_Classification.ipynb     # Complete English analysis
├── data/
│   └── emails.csv                      # Labeled email dataset
├── Email_Spam/
│   ├── email_spam_pipeline.py          # Reusable cleaning functions
│   ├── Email_Spam_Final_Pipeline.pkl   # Trained end-to-end artifact
│   └── deployment/
│       ├── app.py                      # FastAPI service
│       ├── Dockerfile
│       ├── cloudbuild.yaml
│       └── tests/test_api.py
├── requirements.txt
└── README.md
```

## Run the Notebook

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook Email_Spam_Classification.ipynb
```

## Run the API

```bash
pip install -r Email_Spam/deployment/requirements.txt
uvicorn Email_Spam.deployment.app:app --host 0.0.0.0 --port 8080
```

Example request:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Free money! Click here to claim your prize."}'
```

Interactive API documentation is available at `http://localhost:8080/docs`.

## Docker and Cloud Deployment

```bash
docker build -f Email_Spam/deployment/Dockerfile -t email-spam-api .
docker run --rm -p 8080:8080 email-spam-api
```

Google Cloud Run instructions are available in [Email_Spam/deployment/DEPLOYMENT.md](Email_Spam/deployment/DEPLOYMENT.md).

## Limitations

- Email language and spam tactics change over time, so production performance requires monitoring and retraining.
- The model analyzes message text but does not use sender reputation, attachments, or mail-server metadata.
- The dataset is mostly English and should not be treated as representative of all languages or domains.
- The serialized model was created with scikit-learn `1.8.0`; use the pinned deployment environment when loading it.

## Technology Stack

Python, pandas, NumPy, SciPy, scikit-learn, Matplotlib, Seaborn, FastAPI, pytest, Docker, Google Cloud Build, and Google Cloud Run.

