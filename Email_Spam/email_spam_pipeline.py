import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion, FunctionTransformer, Pipeline


def clean_email_text(text):
    """Normalize one raw email while preserving useful semantic tokens."""
    text = str(text).lower()
    text = re.sub(r"^subject:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " html ", text)
    text = re.sub(
        r"\b[\w.+-]+\s*@\s*[\w.-]+\s*\.\s*[a-z]{2,}\b",
        " email ",
        text,
    )
    text = re.sub(r"https?\s*:\s*/\s*/", " url ", text)
    text = re.sub(r"www\s*\.\s*", " url ", text)
    text = re.sub(r"[$€£]\s*\d+(?:[.,]\d+)?", " money ", text)
    text = re.sub(r"\d+", " number ", text)
    text = re.sub(r"[^a-z_\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_email_series(values):
    """Apply the same deterministic cleaning to a Series or list."""
    return pd.Series(values).fillna("").astype(str).map(clean_email_text)


def build_final_pipeline():
    """Create the locked raw-text-to-prediction pipeline."""
    features = FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                max_features=20000,
                sublinear_tf=True,
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=30000,
                sublinear_tf=True,
            ),
        ),
    ])

    model = SGDClassifier(
        alpha=0.00019089171584870895,
        average=False,
        class_weight="balanced",
        loss="hinge",
        penalty="elasticnet",
        max_iter=3000,
        random_state=0,
    )

    return Pipeline([
        (
            "cleaner",
            FunctionTransformer(clean_email_series, validate=False),
        ),
        ("features", features),
        ("model", model),
    ])
