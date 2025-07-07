*# SENTIMENT-ANALYSIS-WITH-NLP

COMPANY : CODTECH IT SOLUTIONS

NAME : G PAVANI

INTERN ID : CT08DK681

DOMAIN : MACHINE LEARNING

DURATION : 8 WEEKS

MENTOR : NEELA SANTHOSH

*DESCRIPTION*

As part of my machine learning internship with CodTech IT Solutions, I performed sentiment analysis with NLP task, which involved performing sentiment analysis on a dataset of customer reviews. The primary objective of this task was to understand and implement Natural Language Processing (NLP) techniques to classify text data (reviews) into positive or negative sentiments using the TF-IDF vectorization method and Logistic Regression model. I successfully completed this task using the Python programming language, Jupyter Notebook, and key Python libraries such as pandas, nltk, sklearn, and matplotlib.

**Tools and Technologies Used :

  Language: Python

  Development Environment: Jupyter Notebook

  Libraries:

  *pandas – for data loading and manipulation

  *nltk (Natural Language Toolkit) – for text preprocessing tasks like stopword removal and tokenization

  *sklearn (scikit-learn) – for implementing TF-IDF vectorization and Logistic Regression model

  *matplotlib & seaborn – for data visualization

  Dataset: A publicly available sentiment-labeled customer reviews dataset (e.g., Amazon, Yelp, or IMDb reviews)

**Step-by-Step Process

1. Importing Libraries :
The first step in my notebook was to import the necessary libraries. These included tools for data handling (pandas), text processing (nltk), and machine learning (sklearn).

2. Loading the Dataset :
I used a CSV file containing customer reviews and corresponding sentiment labels (positive or negative). The data was loaded using pandas.read_csv(), and I previewed the data using .head() and checked the number of entries using .shape.

3. Text Preprocessing :
Text data is usually noisy and requires preprocessing. I performed the following tasks using NLTK and basic Python functions:

Lowercasing – converting all text to lowercase for consistency

Punctuation removal – removing symbols and special characters

Tokenization – splitting reviews into individual words

Stopword removal – filtering out common words like "the", "is", "and"

Lemmatization/Stemming – reducing words to their root form

The cleaned text was stored in a new column for further processing.

4. Vectorization using TF-IDF :
Once the text was cleaned, I used the TF-IDF (Term Frequency-Inverse Document Frequency) technique to convert the textual data into numerical format. TF-IDF is effective in giving importance to words that are unique to a document and penalizing common words. I used TfidfVectorizer from sklearn.feature_extraction.text.

5. Model Building using Logistic Regression :
I split the data into training and testing sets using train_test_split. Then, I trained a Logistic Regression model, which is a widely used algorithm for binary classification problems like sentiment analysis.

6. Model Evaluation :
To evaluate the model, I used metrics such as accuracy, confusion matrix, precision, recall, and F1-score from sklearn.metrics. These metrics gave me a clear understanding of how well my model was performing on unseen test data.

I also visualized the confusion matrix using seaborn.heatmap to better understand false positives and false negatives.

**Challenges and Learnings :
While working on this task, one of the challenges I faced was handling imbalanced data, where positive or negative reviews dominated. I learned about techniques like stratified sampling and resampling to address this. Another key learning was how critical text preprocessing is for NLP-based tasks—poor preprocessing can drastically reduce model performance.

**Conclusion :
This task helped me gain hands-on experience in text classification using traditional machine learning methods. I learned how to clean and process text data, vectorize it using TF-IDF, and apply a Logistic Regression classifier. The performance was satisfactory, with good accuracy, proving that even classical models can provide valuable insights into customer sentiment.

Overall, completing Task 2 helped me understand a real-world application of NLP and machine learning for business insights and decision-making. The experience boosted my confidence in handling text data and preparing it for machine learning tasks.

