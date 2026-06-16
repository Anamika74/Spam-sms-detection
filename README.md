#  SMS Spam Detection System

An end-to-end Machine Learning web application that classifies SMS messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and supervised machine learning techniques.

The project covers the complete ML workflow including data cleaning, exploratory data analysis, text preprocessing, feature engineering, model comparison, and deployment with Streamlit.

---

##  Project Overview

The workflow includes:

* Data Cleaning & Preparation
* Exploratory Data Analysis (EDA)
* Text Preprocessing
* TF-IDF Feature Extraction
* Model Training & Evaluation
* Streamlit Web Application Deployment

---

##  Dataset

Dataset: SMS Spam Collection Dataset

* Total Messages: 5,572
* Classes:

  * Ham (Legitimate Messages)
  * Spam (Unwanted Messages)

Source:
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

---

##  Machine Learning Pipeline

### 1. Data Cleaning

* Removed unnecessary columns
* Handled missing values
* Removed 403 duplicate records
* Fixed encoding issues (`latin-1`)

### 2. Exploratory Data Analysis

Created additional features:

* Number of Characters
* Number of Words
* Number of Sentences

Analyzed differences between Spam and Ham messages using statistical visualizations.

### 3. Text Preprocessing

* Lowercase conversion
* Tokenization
* Removal of special characters
* Stopword removal using NLTK
* Porter Stemmer for word normalization

### 4. Feature Extraction

Used **TF-IDF Vectorization** to transform text into numerical representations.

* Vocabulary Size: 6,708 Features

### 5. Model Training

Several machine learning algorithms were trained and compared:

* Naive Bayes
* Logistic Regression
* Decision Tree
* Random Forest
* Extra Trees
* K-Nearest Neighbors
* Support Vector Machine
* AdaBoost
* Gradient Boosting
* XGBoost
* Bagging Classifier

---

##  Model Performance

| Model               | Accuracy | Precision |
| ------------------- | -------- | --------- |
| Random Forest       | 97.19%   | 1.0000    |
| Naive Bayes         | 95.93%   | 1.0000    |
| KNN                 | 90.03%   | 1.0000    |
| Extra Trees         | 97.29%   | 0.9824    |
| SVC                 | 97.29%   | 0.9741    |
| XGBoost             | 97.48%   | 0.9516    |
| AdaBoost            | 96.13%   | 0.9454    |
| Logistic Regression | 95.16%   | 0.9400    |
| Gradient Boosting   | 95.26%   | 0.9238    |
| Bagging Classifier  | 95.84%   | 0.8625    |
| Decision Tree       | 93.61%   | 0.8461    |

###  Final Model Selection

Although XGBoost achieved the highest overall accuracy, the **Random Forest Classifier** was selected as the production model because it achieved:

* Accuracy: 97.19%
* Precision: 1.0000

For spam detection systems, precision is critical because incorrectly labeling legitimate messages as spam can negatively impact users.

---

##  Project Structure

```text
SMS-Spam-Detection/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── 01_main_file.ipynb
├── requirements.txt
└── README.md
```

### Files Description

| File               | Description                                       |
| ------------------ | ------------------------------------------------- |
| app.py             | Streamlit application       |
| model.pkl          | Trained Random Forest model                       |
| vectorizer.pkl     | TF-IDF vectorizer                                 |
| 01_main_file.ipynb | EDA, preprocessing, model training and evaluation |
| requirements.txt   | Project dependencies                              |

---
