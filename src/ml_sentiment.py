import string
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure required NLTK corpora are present in any deployment (e.g., DigitalOcean)
NLTK_RESOURCES = {
    "stopwords": "corpora/stopwords",
    "punkt": "tokenizers/punkt",
    "wordnet": "corpora/wordnet",
    "vader_lexicon": "sentiment/vader_lexicon",
    "punkt_tab": "tokenizers/punkt_tab",
}


def ensure_nltk_resources():
    for resource, path in NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


ensure_nltk_resources()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "train5.csv"
MODEL_DIR = BASE_DIR / "models"
VECTORIZER_PATH = MODEL_DIR / "count_vectorizer.joblib"
MODEL_PATHS = {
    "naive bayes": MODEL_DIR / "naive_bayes.joblib",
    "svm": MODEL_DIR / "svm.joblib",
}
SCORE_MODEL_PATH = MODEL_DIR / "score_regressor.joblib"
_VADER_ANALYZER = None


def _normalize_model_name(model_name):
    name = str(model_name).strip().lower()
    if name in ('naive bayes', 'naivebayes', 'nb', 'multinomialnb'):
        return "naive bayes"
    if name in ('svm', 'support vector machine', 'support-vector-machine'):
        return "svm"
    raise ValueError(f"Unknown model_name: {model_name}")

def preprocess(text):
    text = str(text).translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    token = word_tokenize(text)
    filtered = []
    for i in token:
        if i not in stop_words:
            filtered.append(lemmatizer.lemmatize(i))
    
    return ' '.join(filtered)
    
def read_file(filename):
    df = pd.read_csv(filename)
    df.columns = ['Sentiment','Text','Score']
    return df 


def train_and_cache_models(data_path=None, force=False):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    data_path = Path(data_path) if data_path else DATA_PATH
    vectorizer_ready = VECTORIZER_PATH.exists()
    models_ready = all(path.exists() for path in MODEL_PATHS.values())
    score_ready = SCORE_MODEL_PATH.exists()

    if not force and vectorizer_ready and models_ready and score_ready:
        return

    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    df = read_file(data_path)
    df['Text'] = df['Text'].astype(str).apply(preprocess)

    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(df['Text'].values)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    y_sentiment = df['Sentiment'].values
    nb_model = MultinomialNB()
    nb_model.fit(X_vec, y_sentiment)
    joblib.dump(nb_model, MODEL_PATHS["naive bayes"])

    svm_model = SVC()
    svm_model.fit(X_vec, y_sentiment)
    joblib.dump(svm_model, MODEL_PATHS["svm"])

    y_score = pd.to_numeric(df['Score'], errors='coerce').fillna(0).values
    score_model = LinearRegression()
    score_model.fit(X_vec, y_score)
    joblib.dump(score_model, SCORE_MODEL_PATH)


def load_cached_vectorizer(train_if_missing=True, data_path=None):
    if not VECTORIZER_PATH.exists():
        if train_if_missing:
            train_and_cache_models(data_path=data_path)
        else:
            raise FileNotFoundError(f"Missing cached vectorizer: {VECTORIZER_PATH}")
    return joblib.load(VECTORIZER_PATH)


def load_cached_model(model_name, train_if_missing=True, data_path=None):
    normalized = _normalize_model_name(model_name)
    model_path = MODEL_PATHS[normalized]
    if not model_path.exists():
        if train_if_missing:
            train_and_cache_models(data_path=data_path)
        else:
            raise FileNotFoundError(f"Missing cached model: {model_path}")
    model = joblib.load(model_path)
    vectorizer = load_cached_vectorizer(train_if_missing=train_if_missing, data_path=data_path)
    return vectorizer, model


def predict_cached(X_test, model_name, train_if_missing=True, data_path=None):
    vectorizer, model = load_cached_model(model_name, train_if_missing=train_if_missing, data_path=data_path)
    X_vec = vectorizer.transform(X_test)
    return model.predict(X_vec)


def load_cached_score_model(train_if_missing=True, data_path=None):
    if not SCORE_MODEL_PATH.exists():
        if train_if_missing:
            train_and_cache_models(data_path=data_path)
        else:
            raise FileNotFoundError(f"Missing cached score model: {SCORE_MODEL_PATH}")
    model = joblib.load(SCORE_MODEL_PATH)
    vectorizer = load_cached_vectorizer(train_if_missing=train_if_missing, data_path=data_path)
    return vectorizer, model


def predict_score_cached(X_test, train_if_missing=True, data_path=None):
    vectorizer, model = load_cached_score_model(train_if_missing=train_if_missing, data_path=data_path)
    X_vec = vectorizer.transform(X_test)
    return model.predict(X_vec)

def emotion_score(X_train, y_train, X_test):
    vector = CountVectorizer()
    X_train2 = vector.fit_transform(X_train)
    X_test2 = vector.transform(X_test)

    model = LinearRegression()
    model.fit(X_train2,y_train)

    score = model.predict(X_test2)
    return score

def prebuilt_model(X_test):
    analyzer = get_vader_analyzer()
    predictions = []
    for i in X_test:
        score = analyzer.polarity_scores(i)
        score1 = score['compound']
        if score1 >= 0.2:
            predictions.append('positive')
        elif score1 <= -0.2:
            predictions.append('negative')
        else:
            predictions.append('neutral')
    return np.array(predictions)


def get_vader_analyzer():
    global _VADER_ANALYZER
    if _VADER_ANALYZER is None:
        _VADER_ANALYZER = SentimentIntensityAnalyzer()
    return _VADER_ANALYZER


def vader_score(text):
    analyzer = get_vader_analyzer()
    return analyzer.polarity_scores(text)['compound']

def evaluate_model(X,y,model,type=0,k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []
    precisions = []
    recalls = []
    confusions = []
    f1_scores = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if type == 0:
            pred = my_model(X_train, y_train, X_test, model)
        elif type == 1:
            pred = prebuilt_model(X_test)

        accuracy = accuracy_score(y_test,pred)
        precision = precision_score(y_test,pred, average=None)
        recall = recall_score(y_test,pred,average=None)
        confusion = confusion_matrix(y_test,pred)
        f1 = f1_score(y_test, pred, average=None)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        confusions.append(confusion)
        f1_scores.append(f1)

    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions, axis=0)
    avg_recall = np.mean(recalls, axis=0)
    sum_confusion = np.sum(confusions, axis=0)
    avg_f1 = np.mean(f1_scores, axis=0) 
    return avg_accuracy, avg_precision, avg_recall, sum_confusion, avg_f1

def my_model(X_train, y_train, X_test, model_name):
    vector = CountVectorizer()
    X_train2 = vector.fit_transform(X_train)
    X_test2 = vector.transform(X_test)

    normalized = _normalize_model_name(model_name)
    if normalized == "naive bayes":
        clf = MultinomialNB()
    elif normalized == "svm":
        clf = SVC()
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    clf.fit(X_train2, y_train)
    pred = clf.predict(X_test2)
    return pred


