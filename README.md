# Sentiment-Analysis-of-Movie-Reviews-By-Applying-TF-IDF

## Introduction
Sentiment analysis, also known as opinion mining, is a natural language processing task that aims to determine the sentiment expressed in a piece of text. In the context of movie reviews, sentiment analysis helps to classify reviews as positive, negative, or neutral. This report focuses on leveraging the TF-IDF (Term Frequency-Inverse Document Frequency) technique to train machine learning models that can predict the sentiment of movie reviews sourced from the "Sentiment Analysis on Movie Reviews" Kaggle competition dataset.

## Aim
The primary aim of this study is to apply the TF-IDF technique to preprocess and vectorize movie reviews data and subsequently train various machine learning models to analyze and predict the sentiment of these reviews. The goals include:

Exploring and visualizing the dataset to gain insights into the distribution of sentiments.
Implementing the TF-IDF technique to transform textual data into numerical format suitable for machine learning algorithms.
Training baseline models and evaluating their performance in terms of accuracy.
Fine-tuning and comparing different machine learning models to identify the most effective approach for sentiment classification.
## Methodology
Data Collection and Exploration
The dataset was downloaded from the Kaggle competition page and consists of labeled movie reviews in a tab-separated values (TSV) format. The initial exploration involved:

Loading the training, test, and submission files using Pandas.
Analyzing the distribution of sentiments in the dataset and visualizing the results through bar graphs and word clouds.
Implementation of TF-IDF Technique
The TF-IDF technique was employed to convert the text data into a format suitable for machine learning. This involved:

Tokenizing the phrases and applying stemming to reduce words to their root forms.
Removing stop words to enhance the quality of the textual data.
Configuring the TfidfVectorizer to learn the vocabulary and transform both training and test data into numerical representations.
Model Training
A baseline model using Logistic Regression was trained on the transformed training data. The dataset was split into training and validation sets, allowing for performance evaluation using accuracy scores. Further, two additional models—a Random Forest Classifier and a Gradient Boosting Classifier—were trained and evaluated. The performance of each model was measured using accuracy metrics on both the training and validation datasets.

Submission to Kaggle
After training the models, predictions were made on the test dataset, and the results were submitted to Kaggle for evaluation. Screenshots of the submission scores were captured for documentation.

## Results
The following findings were observed during the analysis:

The dataset contained 156,060 training samples, with sentiment class distribution showing that neutral sentiment (Sentiment 2) was the most prevalent, comprising approximately 51% of the data.
The TF-IDF vectorization resulted in 2,000 features extracted from the training data.
The baseline Logistic Regression model achieved an accuracy of approximately 63.92% on the training set and around 57.93% on the validation set.
The Random Forest Classifier demonstrated a higher training accuracy of 79.03%, but its performance on the validation set was relatively lower at 56.10%.
The Gradient Boosting Classifier exhibited lower accuracy, with scores of 56.04% on the training set and 51.91% on the validation set.
## Conclusion
This study successfully implemented the TF-IDF technique to preprocess movie reviews for sentiment analysis. Although the baseline Logistic Regression model provided decent accuracy, the Random Forest Classifier yielded the highest training accuracy among the models tested, despite underperforming on the validation set. Further tuning of hyperparameters and exploring additional machine learning algorithms could enhance the model's performance. This report underscores the importance of proper text preprocessing and model selection in achieving effective sentiment classification in natural language processing tasks.
