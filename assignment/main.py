#!/usr/bin/env python3
"""Assignment: Data mining tasks using Dataset/adult.csv

Run: python3 assignment/main.py
"""
#!/usr/bin/env python3
"""Full assignment implementation — restored as second commit.

This file contains the full set of data mining tasks required by the
assignment (error detection/fixing, outlier detection, regressions,
classifiers, clustering, and association rules).
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import difflib
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from mlxtend.frequent_patterns import apriori, association_rules


DATA_PATH = "Dataset/adult.csv"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df


def _col_name_map(df):
    m = {}
    for c in df.columns:
        key = ''.join([ch if ch.isalnum() else '_' for ch in c.lower()]).strip('_')
        m[key] = c
    return m


def _find_col(df, name):
    key = ''.join([ch if ch.isalnum() else '_' for ch in name.lower()]).strip('_')
    m = _col_name_map(df)
    if key in m:
        return m[key]
    if name in df.columns:
        return name
    candidates = difflib.get_close_matches(key, list(m.keys()), n=1, cutoff=0.6)
    if candidates:
        chosen = m[candidates[0]]
        print(f"Warning: using column '{chosen}' for requested '{name}' (fuzzy match)")
        return chosen
    raise KeyError(f"Column matching '{name}' not found in dataframe")


def detect_and_fix_errors(df):
    df = df.copy()
    df.replace("?", np.nan, inplace=True)
    missing = df.isnull().sum()
    print("Missing values per column:\n", missing[missing > 0])
    before = len(df)
    df.dropna(inplace=True)
    after = len(df)
    print(f"Dropped {before-after} rows with missing values.")
    return df


def detect_outliers(df, numeric_cols=None, z_thresh=3.0):
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_idx = set()
    for c in numeric_cols:
        col = df[c].astype(float)
        z = (col - col.mean()) / col.std(ddof=0)
        idx = df.index[np.abs(z) > z_thresh].tolist()
        if idx:
            print(f"Outliers in {c}: {len(idx)}")
            outlier_idx.update(idx)
    print(f"Total rows with outlier values: {len(outlier_idx)}")
    return sorted(outlier_idx)


def linear_regression_single(df):
    age_col = _find_col(df, "age")
    hours_col = _find_col(df, "hours-per-week")
    X = df[[age_col]].values
    y = df[hours_col].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"LinearRegression R^2 (hours-per-week ~ age): {score:.4f}")
    return model


def multiple_linear_regression(df):
    feat_names = ["age", "education-num", "capital-gain", "capital-loss"]
    features = [_find_col(df, n) for n in feat_names]
    X = df[features].astype(float)
    y = df[_find_col(df, "hours-per-week")].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print(f"Multiple Linear Regression R^2: {lr.score(X_test, y_test):.4f}")
    return lr


def polynomial_regression(df, degree=2):
    age_col = _find_col(df, "age")
    hours_col = _find_col(df, "hours-per-week")
    X = df[[age_col]].values
    y = df[hours_col].values
    pf = PolynomialFeatures(degree=degree, include_bias=False)
    Xp = pf.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(Xp, y, test_size=0.25, random_state=0)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print(f"Polynomial degree {degree} R^2: {lr.score(X_test, y_test):.4f}")
    return lr


def prepare_classification(df):
    df = df.copy()
    income_col = _find_col(df, "income")
    df[income_col] = df[income_col].str.strip()
    y = (df[income_col] == ">50K").astype(int)
    num_names = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    num_cols = [_find_col(df, n) for n in num_names]
    X = df[num_cols].astype(float)
    return X, y


def logistic_regression(df):
    X, y = prepare_classification(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_s, y_train)
    preds = clf.predict(X_test_s)
    probs = clf.predict_proba(X_test_s)[:, 1]
    print_eval(y_test, preds, probs)
    return clf


def decision_tree_classifier(df):
    X, y = prepare_classification(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)
    preds = dt.predict(X_test)
    probs = dt.predict_proba(X_test)[:, 1]
    print_eval(y_test, preds, probs)
    return dt


def apriori_algorithm(df, min_support=0.05):
    cats = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex"]
    actual = []
    for c in cats:
        try:
            actual.append(_find_col(df, c))
        except KeyError:
            continue
    if not actual:
        actual = df.select_dtypes(include=[object]).columns.tolist()
    dfc = df[actual].apply(lambda x: x.str.strip())
    df_ohe = pd.get_dummies(dfc)
    frequent = apriori(df_ohe, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent, metric="confidence", min_threshold=0.6)
    print(f"Found {len(frequent)} frequent itemsets and {len(rules)} rules.")
    return frequent, rules


def print_eval(y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else float("nan")
    print("Confusion matrix:\n", cm)
    print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {auc:.4f}")


def knn_classifier(df, k=5):
    X, y = prepare_classification(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_s, y_train)
    preds = knn.predict(X_test_s)
    probs = knn.predict_proba(X_test_s)[:, 1]
    print_eval(y_test, preds, probs)
    return knn


def kmeans_clustering(df, n_clusters=3):
    cols = [_find_col(df, n) for n in ["age", "education-num", "hours-per-week"]]
    X = df[cols].astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=0)
    km.fit(Xs)
    labels = km.labels_
    print(f"KMeans inertia: {km.inertia_:.2f}")
    print(pd.Series(labels).value_counts())
    return km


def naive_bayes_classifier(df):
    X, y = prepare_classification(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    nb = GaussianNB()
    nb.fit(X_train_s, y_train)
    preds = nb.predict(X_test_s)
    probs = nb.predict_proba(X_test_s)[:, 1]
    print_eval(y_test, preds, probs)
    return nb


def main():
    print("Loading data from", DATA_PATH)
    df = load_data()
    print("Rows, cols:", df.shape)
    df = detect_and_fix_errors(df)
    detect_outliers(df)
    linear_regression_single(df)
    multiple_linear_regression(df)
    polynomial_regression(df, degree=2)
    print("\n-- Logistic Regression --")
    logistic_regression(df)
    print("\n-- Decision Tree --")
    decision_tree_classifier(df)
    print("\n-- Apriori (association rules) --")
    apriori_algorithm(df)
    print("\n-- KNN --")
    knn_classifier(df)
    print("\n-- KMeans clustering --")
    kmeans_clustering(df)
    print("\n-- Naive Bayes --")
    naive_bayes_classifier(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
        return name
