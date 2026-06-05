#!/usr/bin/env python3
"""Assignment: Data mining tasks using Dataset/adult.csv

Run: python3 assignment/main.py
"""
#!/usr/bin/env python3
"""Minimal initial version of the assignment script.

This initial commit contains just loading, basic cleaning, and a single
linear regression example. Later commits expand functionality.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


DATA_PATH = "Dataset/adult.csv"


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def detect_and_fix_errors(df):
    df = df.copy()
    df.replace("?", np.nan, inplace=True)
    print("Missing values per column:\n", df.isnull().sum()[df.isnull().sum() > 0])
    df.dropna(inplace=True)
    return df


def linear_regression_single(df):
    # Simple regression: hours-per-week ~ age
    X = df[["age"]].astype(float).values
    y = df["hours-per-week"].astype(float).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"LinearRegression R^2: {model.score(X_test, y_test):.4f}")


def main():
    df = load_data()
    print("Rows, cols:", df.shape)
    df = detect_and_fix_errors(df)
    linear_regression_single(df)


if __name__ == "__main__":
    main()
        return name
