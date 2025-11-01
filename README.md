# THEORY

# PROBLEM STATEMENT

The proliferation of unsolicited and fraudulent messages through SMS channels poses a major challenge for mobile communication systems. Manual filtering is inefficient and infeasible at scale. Therefore, the problem addressed in this project is:
“To develop an automated machine learning-based system capable of detecting and classifying SMS messages as either spam or legitimate (ham) based on their textual content.”

# OBJECTIVE

The specific objectives of the project are:

1) Data Acquisition and Preprocessing: Import and clean the SMS dataset by removing inconsistencies and irrelevant characters.

2) Feature Engineering: Convert raw text into vectorized numerical features suitable for machine learning models.

3) Model Development: Train and optimize a supervised classifier to distinguish spam from ham messages.

4) Model Evaluation: Measure model performance using quantitative metrics such as confusion matrix, accuracy, precision, recall, and F1-score.

5) Deployment: Implement a real-time classification system using the serialized model (model.pkl) integrated with an interactive front-end (app.py).

# DATASET AND FEATURES

DATASET:-

1) Name: SMS Spam Collection Dataset

2) Source: The dataset included (spam.csv) corresponds to the UCI Machine Learning Repository SMS Spam Collection Dataset.

3) Size: 5,574 labeled SMS messages

4) Attributes:-

Label: Binary categorical variable indicating message type — ham or spam.

Message: The raw text content of the SMS message.

Distribution: ~86% ham, ~14% spam (class imbalance considered during training).

5) Preprocessing Steps:-

Lowercasing

Removal of punctuation, digits, and special symbols

Tokenization

Stop-word removal

Optional stemming/lemmatization

FEATURES:-

After preprocessing, the text data is vectorized using NLP-based feature representations:

  FEATURE TYPE    :                        DESCRIPTION                                     :  TOOL/METHOD USED
  
Token frequency   :  Basic bag-of-words representation (count of terms in each message)    :  CountVectorizer

TF-IDF weighting  :  Weighted representation emphasizing rare but important terms          :  TfidfVectorizer

n-gram features   :  Captures token sequences (e.g., bigrams) to preserve partial context  :  TfidfVectorizer(ngram_range=(1,2))
