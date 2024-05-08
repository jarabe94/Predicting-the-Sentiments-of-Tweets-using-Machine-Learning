# Predicting the Sentiments of Tweets using Machine Learning

**Introduction**

Social media sites like Twitter/ X have become essential components of our everyday lives, acting as centers for communication and information in real time. Sentiment analysis provides information on trends, opinions, and attitudes about a wide range of subjects and areas. 

In this project, we share our joint endeavor to examine the opinions conveyed in a tweets dataset and build a robust sentiment analysis pipeline to offer invaluable insights for understanding sentiment trends on Twitter/ X. We are going to solve the challenges of sentiment classification of tweets by developing several machine learning models to classify tweets into positive and negative sentiments.

Relevance:

- Crisis Management: Our project can help organizations monitor social media for negative sentiments and respond accordingly in the event of a crisis.
  
- Reputation Management: Our project can assist organizations in managing their reputation on social media by addressing negative reviews.

- Understanding Customer Feedback: Our project can assist organizations in managing their reputation on social media in the event of negative reviews.

- Market Research: Our project can aid digital marketers in understanding customer’s behavior & preferences, thereby developing targeted campaigns.

- Political Analysis: Our project can help political entities understand public opinion by monitoring social media, thereby tailoring their narrative accordingly.


**Dataset Characteristics**

The dataset consists of six columns. Based on the initial examination, here's a brief overview:

- label: Appears to be the target variable with integer values, likely indicating sentiment (e.g., 0 for negative, 1 for positive).

- tweet_id: Contains the tweet ID (also an integer).

- timestamp: Contains timestamps in a readable format.

- query: Has a constant value "NO_QUERY"; its purpose is unclear with only a reference to the presence of query (Lyx). However, it does not seem relevant for sentiment analysis.

- user_id: Contains usernames.

- text: Contains the text data (tweet), which is what we're most interested in for sentiment analysis.

The data types are as follows:

Integer for columns label and tweet_id, and object (string) for columns timestamp to user_id. Given this structure, our target variable for sentiment analysis is in Column label, and the feature available for prediction is the text data in Column text.

**Exploratory Data Analysis**

The following steps were included:

- Visualize the Distribution of the Target Variable: This helped us assess the balance or imbalance of sentiment within our dataset.
  
- Explore Text Data: Identified common words and phrases to understand the general sentiment and topics by generating word clouds. 

- Handle Missing Values: Checked for and handle any missing values in the dataset, especially in the text data and the target variable.

- Visualize Text Length Distribution: Understanding the length of the tweets gave us insights into the dataset's characteristics.

EDA Insights:

Sentiment Distribution:

- The distribution of sentiments was balanced with two distinct classes, indicating positive and negative sentiments.

- Each class had roughly the same number of entries, suggesting that the dataset is well-suited for training classification models without the need for sampling techniques to address class imbalance.
  
Missing Values:

- There were no missing values in any of the columns, meaning there was no need for imputation or removal of missing data.
  
Word Cloud:

- The word cloud suggested a mix of positive and negative words, with positive words like "love," "thank," and "good" appearing prominently, as well as negative words such as "miss" and "hate."

- Common terms related to social media interactions like "twitter," "twitpic", and "facebook" were also visible, which was expected given the context of the data.

- Words related to daily life and personal experiences, such as "work", "going", "today", "back", and "time", indicated the personal nature of the text content.

Text Length Statistics:

- The average length of a text entry was around 74 characters, with a standard deviation of about 36 characters, suggesting moderate variance in text length.

- The minimum text length was very short at only 6 characters, and the maximum length was 374 characters, which was close to the historical 280-character limit of Twitter, suggesting that the text data was sourced from a platform with a character limit.

- The 25th percentile was at 44 characters, the median (50th percentile) was at 69 characters, and the 75th percentile was at 104 characters, indicating a right-skewed distribution where most texts were shorter rather than longer.

**Data Preprocessing** 

Data preprocessing involves cleaning and preparing the text data for modeling. Here are the steps followed for data preprocessing:

- Define emojis and chat words dictionaries: Emojis and chat words dictionaries were used in the preprocessing functions.

- Remove User Mentions and Hashtags: User mentions (e.g., "@username") and hashtags (e.g., "#example") are common in social media data but do not contribute to the analysis.

- Replace Emojis: Emojis can convey sentiments or emotions, but they can be challenging for machine learning models to interpret directly, therefore we replaced them with the word of the sentiment they represent.

- Replace Chatwords: Chatwords or internet slangs (e.g., "lol", "brb") may not be recognized by standard language processing tools.

- Remove Punctuation: Punctuation marks such as periods, commas, and exclamation points may not carry significant meaning for some tasks and can be removed from the text using string manipulation functions or regular expressions.

- Remove Stopwords: Stopwords are common words like "the", "is", "and", etc., that occur frequently in text but often do not contribute much to the overall meaning.

- Remove URL: URLs often appear in text data but may not be relevant to the analysis.

- Lemmatization: Reduced words to their root form. Lemmatization is typically more useful than Stemming as it converts words to their meaningful base form.

- Vectorization: Converted text to a numerical format that machine learning models could understand using TF-IDF vectorization.

- Split the Data: Divided the dataset into training (80%) and testing (20%) sets.

**Model Building and Evaluation**

In this section, we built and compared seven models to determine the most effective predictor of tweet sentiments.

The first six models were supervised, meaning they were trained using labeled data, where each tweet was associated with a sentiment label.

The seventh model was unsupervised, meaning it did not rely on labeled data. Instead, it analyzed the structure and patterns within the data to identify sentiment clusters or themes without explicit supervision.

Supervised models:

- Logistic Regression: Versatile model for binary classification, it's a good starting point for text sentiment analysis.

- Naive Bayes: Probabilistic model, often performs well in text classification due to its assumption of independence among features.

- Linear Support Vector Machine (SVC): Classification algorithm efficient for linearly separable data in high-dimensional spaces.

- Random Forest: An ensemble method that can capture non-linear patterns by combining decision trees.

- Stochastic Gradient Descent Classifier (SGDC) with Logistic Regression: SGDC is an efficient optimization algorithm used to train logistic regression models.

- Artificial Neural Network (ANN): A versatile model inspired by the human brain's structure and capable of capturing intricate patterns in text.

Unsupervised model:

- VADER: Stands for "Valence Aware Dictionary and sEntiment Reasoner". Lexicon and rule-based sentiment analysis tool, suitable for sentiment analysis on social media text.

For each model, we instantiated the model, trained it on the training set, and predicted sentiments on the test set.

After comparing all the models, we concluded that the Artificial Neural Network (ANN) was the top performer model with an accuracy of 78.84%. Additionally, it achieved the highest score for precision and recall, both of 78.84%. This indicates that the ANN model not only had strong overall predictive accuracy but also effectively balanced the trade-off between precision (the proportion of correctly predicted positive instances among all predicted positives) and recall (the proportion of correctly predicted positive instances among all actual positives), making it the most reliable model for sentiment analysis tasks.

While VADER achieved the lowest performance with an accuracy of 64.38%, it's essential to note that unsupervised models often exhibit lower accuracy compared to supervised models. This discrepancy can be attributed to the fact that unsupervised models, like VADER, operate on unlabeled data and must autonomously discover the underlying structure of it.

**Model Optimization**

Model optimization involves selecting the right parameters for our models to improve their performance. This is usually done through a process known as hyperparameter tuning, where we can search through a predefined space of hyperparameters to find the combination that performs best on our validation set. Common strategies include Grid Search and Random Search.

We focused on optimizing the second and third best performing models, Logistic Regression and Linear Support Vector Machine, respectively. Our approach to model optimization involved:

- For Logistic Regression: We optimized the regularization strength (C) using 3-fold cross-validation and l2 penalty.

- For Linear SVC: We tuned the regularization parameter (C) using a randomized search approach.

After optimizing the models, we did not achieve any significant improvement in accuracy. However, it's worth noting that training such optimized models required more computation time and increased complexity.

**Sentiment Prediction**

Since the ANN model exhibited the highest accuracy, we chose it as the base model to conduct sentiment predictions, and subsequently stored it in a pickle file.

**Conclusion**

This project successfully built a robust sentiment analysis pipeline and gained valuable insights into machine learning techniques for text classification.

We explored various supervised and unsupervised machine learning models to predict sentiments in tweets. The Artificial Neural Network (ANN) emerged as the top performer, achieving the highest accuracy among the models tested and optimized. For model selection, it is also essential to consider the computational costs and complexities associated with the models.

Sentiment analysis on tweets holds significant importance in understanding public opinion, market trends, and brand perception in real-time.

Moving forward, further experimentation with different feature engineering techniques and model architectures could potentially enhance predictive performance. Furthermore, integrating the model into a web application or API can support real-time sentiment prediction.


