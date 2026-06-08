# Data Mining Assignment Presentation

## Project Title
**Data Mining Analysis on the UCI Adult Dataset**

## Student Information
**Author:** Kazi Neyamul Hasan  
**Student ID:** 0112230359

## Document Purpose
This document is a formal presentation handout and submission companion for the assignment implemented in `assignment/main.py` and documented in `Data_Mining_Assignment.ipynb`.

---

## 1. Executive Summary

This assignment analyzes the UCI Adult dataset using a complete data mining workflow. The project includes data cleaning, outlier detection, regression analysis, classification, clustering, and association rule mining. The objective is to demonstrate how a real dataset can be transformed into meaningful insights using Python and standard machine learning libraries.

---

## 2. Assignment Overview

The project uses the dataset stored at `Dataset/adult.csv`. The dataset contains demographic and employment-related attributes such as:

- age
- workclass
- education
- education-num
- marital-status
- occupation
- relationship
- race
- sex
- capital-gain
- capital-loss
- hours-per-week
- income

The main objective is to analyze the dataset with different data mining methods and compare how each technique behaves on a real-world classification problem.

---

## 3. Tools and Libraries Used

The project uses these Python libraries:

- `pandas` for loading and cleaning data
- `numpy` for numerical operations
- `scikit-learn` for machine learning models
- `mlxtend` for Apriori and association rules
- `matplotlib` and `seaborn` are listed in requirements for visualization support

The code is designed to run with:

```bash
python3 assignment/main.py
```

The dependencies are installed with:

```bash
pip install -r requirements.txt
```

---

## 4. High-Level Workflow

The script follows this pipeline:

1. Load the dataset
2. Clean missing values
3. Detect outliers
4. Run regression models
5. Run classification models
6. Run clustering
7. Mine association rules
8. Print evaluation results

This workflow shows the full process from raw data to final analytical output.

---

## 5. Dataset Cleaning and Error Handling

### What the code does

The function `detect_and_fix_errors(df)` performs basic data cleaning:

- Replaces `?` values with `NaN`
- Displays missing values per column
- Drops rows with missing values

### Why this matters

Real-world datasets often contain incomplete or noisy values. Cleaning is necessary before applying machine learning models because most algorithms require valid numeric or categorical input.

### Presentation note

Explain that missing values were converted into standard missing markers and incomplete rows were removed to keep the modeling process consistent and valid.

---

## 6. Outlier Detection

### What the code does

The function `detect_outliers(df, numeric_cols=None, z_thresh=3.0)` checks numeric columns using the z-score method.

- It finds values far away from the mean
- It marks values with absolute z-score greater than 3
- It prints how many outliers exist in each numeric column and the total number of rows affected

### Why this matters

Outliers can distort statistical analysis and reduce model quality, especially for regression and clustering.

### Presentation note

Explain that z-score based detection was used to identify unusually large or small numeric values that could influence regression and clustering results.

---

## 7. Regression Analysis

The assignment includes three regression methods that examine the relationship between age, working hours, and other numeric attributes.

### 7.1 Simple Linear Regression

Function: `linear_regression_single(df)`

- Predicts `hours-per-week` using `age`
- Uses `train_test_split` to divide the data
- Trains a `LinearRegression` model
- Prints the $R^2$ score on the test set

### Presentation note

State that simple linear regression tests whether a single variable, age, can explain variation in weekly working hours.

### 7.2 Multiple Linear Regression

Function: `multiple_linear_regression(df)`

- Uses multiple features:
  - age
  - education-num
  - capital-gain
  - capital-loss
- Predicts `hours-per-week`
- Reports test-set $R^2$

### Presentation note

State that multiple linear regression improves the analysis by using several related features together.

### 7.3 Polynomial Regression

Function: `polynomial_regression(df, degree=2)`

- Uses `age` as input
- Expands the input using `PolynomialFeatures`
- Fits another `LinearRegression` model
- Evaluates it with $R^2$

### Presentation note

State that polynomial regression was included to capture possible non-linear behavior in the data.

---

## 8. Classification Analysis

The classification part predicts whether a person earns more than `50K`.

### Target variable

The code converts the income column into a binary target:

- `1` if income is `>50K`
- `0` otherwise

### Features used

The models use these numeric features:

- age
- education-num
- capital-gain
- capital-loss
- hours-per-week

### Why these features matter

These variables are commonly related to income level and are suitable for classification because they are numeric and easy to standardize.

---

## 9. Logistic Regression

Function: `logistic_regression(df)`

### What it does

- Splits the data into training and testing sets
- Standardizes the features using `StandardScaler`
- Trains a `LogisticRegression` model
- Predicts class labels and probabilities
- Prints:
  - confusion matrix
  - precision
  - recall
  - F1 score
  - ROC-AUC

### Presentation note

Explain that logistic regression serves as a baseline binary classifier for predicting whether income exceeds 50K.

### How to explain the metrics

- **Confusion matrix**: shows correct and incorrect predictions
- **Precision**: how many predicted positive cases are actually positive
- **Recall**: how many actual positive cases are found
- **F1 score**: balance between precision and recall
- **ROC-AUC**: overall quality of probability ranking

---

## 10. Decision Tree Classification

Function: `decision_tree_classifier(df)`

### What it does

- Trains a `DecisionTreeClassifier`
- Predicts income class on the test set
- Prints the same evaluation metrics

### Presentation note

Explain that decision trees are easy to interpret and show how feature-based splits lead to a classification decision.

---

## 11. K-Nearest Neighbors

Function: `knn_classifier(df, k=5)`

### What it does

- Standardizes the input features
- Uses `KNeighborsClassifier`
- Predicts income class based on the nearest neighbors
- Prints classification metrics

### Presentation note

Explain that KNN uses distance between records, so feature scaling is necessary before classification.

---

## 12. Naive Bayes

Function: `naive_bayes_classifier(df)`

### What it does

- Standardizes the numeric features
- Trains a `GaussianNB` model
- Predicts income class
- Prints classification metrics

### Presentation note

Explain that Naive Bayes is a fast probabilistic classifier and works well as a baseline model.

---

## 13. Clustering with K-Means

Function: `kmeans_clustering(df, n_clusters=3)`

### What it does

- Uses the features:
  - age
  - education-num
  - hours-per-week
- Standardizes them
- Applies `KMeans` with 3 clusters
- Prints the inertia value and cluster counts

### Presentation note

Explain that K-Means is unsupervised and groups similar records without using the income label.

### Why this is useful

Clustering helps discover hidden structure in the dataset even when no target label is used.

### Important term

- **Inertia**: the total within-cluster distance, lower values usually mean tighter clusters

---

## 14. Association Rule Mining

Function: `apriori_algorithm(df, min_support=0.05)`

### What it does

- Selects categorical columns such as:
  - workclass
  - education
  - marital-status
  - occupation
  - relationship
  - race
  - sex
- Strips spaces from values
- Converts categories into one-hot encoded columns
- Uses the Apriori algorithm to find frequent itemsets
- Generates association rules with confidence threshold 0.6

### Presentation note

Explain that Apriori identifies frequent patterns and helps discover relationships among categorical attributes.

### Important terms

- **Support**: how often an itemset appears in the dataset
- **Confidence**: how often the rule is correct when the left-hand side occurs
- **Frequent itemset**: a combination of items that appears often enough to be interesting

---

## 15. Why the Helper Function `_find_col()` Is Important

The script includes a flexible column-matching helper named `_find_col(df, name)`.

### What it does

- Normalizes column names
- Handles small formatting differences
- Uses fuzzy matching if needed

### Why it matters

Datasets sometimes have column names with different formatting, such as `hours-per-week` versus `hours_per_week`. This helper makes the code more robust.

### Presentation note

Explain that the helper function makes the code robust to minor differences in column naming.

---

## 16. Main Execution Flow

The `main()` function runs the complete workflow in this order:

1. Load data
2. Print dataset shape
3. Clean data
4. Detect outliers
5. Run simple linear regression
6. Run multiple linear regression
7. Run polynomial regression
8. Run logistic regression
9. Run decision tree classification
10. Run Apriori association rules
11. Run KNN classification
12. Run K-Means clustering
13. Run Naive Bayes classification
14. Print `Done.`

### Presentation note

Explain that the main function runs the entire pipeline from loading data to final model output.

---

## 17. Recommended Presentation Order

Present the project in this order:

1. Introduction to the assignment
2. Dataset description
3. Why cleaning is needed
4. Outlier detection
5. Regression models
6. Classification models
7. Clustering
8. Association rules
9. Final conclusion

This order is easy for the audience to follow because it moves from preprocessing to advanced analysis.


---

## 18. Closing Summary

This assignment demonstrates the full data mining pipeline on the Adult dataset. The workflow includes data cleaning, outlier detection, regression, classification, clustering, and association rule mining. The project shows how different data mining techniques can be applied to the same dataset for different analytical goals.

---

## 19. Strengths of the Project

- Covers multiple data mining techniques in one script
- Uses a real-world dataset
- Includes both supervised and unsupervised learning
- Produces evaluation metrics for comparison
- Uses reusable helper functions for cleaner code

---

## 20. Limitations

- Missing values are handled by dropping rows, which can reduce data size
- The models use only numeric features for classification
- The clustering uses only three features
- Apriori rules depend heavily on chosen support and confidence thresholds
- The notebook/script does not include advanced hyperparameter tuning

These limitations are normal and can be mentioned to show critical understanding.

---

## 21. Final Summary

This project shows how to take a real dataset from raw form to meaningful analysis using data cleaning, outlier detection, regression, classification, clustering, and association rule mining. The code is organized as a complete data mining pipeline and can be executed from one main script.

---

## Short Summary

> My assignment analyzes the Adult dataset using several data mining techniques. I first cleaned missing values and checked for outliers. Then I applied regression models to study relationships, classification models to predict income, K-Means for clustering, and Apriori for association rules. The goal was to demonstrate a full data mining workflow using Python and scikit-learn.

---
