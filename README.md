# SENTIMENT-ANALYSIS

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: SIVAKARTHICK B
I
*INTERN ID*: : CT08FYO

*DOMAIN*: MACHINE LEARNING

*DURATION*: 4 WEEEKS

*MENTOR*: NEELA SANTOSH


**Sentiment Analysis on Amazon Reviews Using Logistic Regression**

### Introduction
Sentiment analysis is a widely used Natural Language Processing (NLP) technique that helps in determining the polarity of textual data, typically classifying sentiments as positive, negative, or neutral. In this project, sentiment analysis is performed on Amazon product reviews, where customer feedback is categorized into positive and negative sentiments based on their review scores. The project leverages machine learning techniques, specifically Logistic Regression, to predict sentiment from textual data.

### Tools and Technologies Used
The project utilizes a combination of Python libraries and machine learning frameworks to preprocess, analyze, and classify textual data efficiently. The key tools used in this project include:

- **Pandas:** Used for data manipulation and analysis, including reading CSV files, handling missing values, and transforming datasets.
- **NumPy:** Supports numerical operations such as array transformations and mathematical computations.
- **Matplotlib & Seaborn:** Used for data visualization, including plotting confusion matrices to analyze model performance.
- **NLTK (Natural Language Toolkit):** A library for text processing that includes stopword removal, tokenization, and stemming.
- **Scikit-learn:** Provides essential machine learning algorithms, including TF-IDF vectorization, logistic regression, and evaluation metrics.
- **Regular Expressions (re module):** Used to clean text by removing special characters, numbers, and punctuation.

### Methodology
The project follows a structured approach to transform raw text data into meaningful insights using machine learning techniques. Below are the key steps undertaken:

1. **Data Collection and Preparation:**
   - The dataset containing Amazon reviews and corresponding scores is loaded using Pandas.
   - The dataset is filtered to retain only the relevant columns (review text and review score).

2. **Data Preprocessing:**
   - Reviews are converted to lowercase to maintain uniformity.
   - Numerical digits and punctuation marks are removed using regular expressions.
   - Stopwords (commonly used words that do not contribute significantly to sentiment classification, such as "the," "is," "and") are eliminated using the NLTK library.
   - Text is tokenized and cleaned to retain only meaningful words.

3. **Label Encoding:**
   - The review score is mapped into binary categories:
     - Scores **4 and 5** are labeled as "Positive" (1).
     - Scores **1, 2, and 3** are labeled as "Negative" (0).

4. **Feature Extraction:**
   - The **TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer** from Scikit-learn is used to convert textual data into numerical representations that can be fed into a machine learning model.
   - A maximum of 5000 features is selected to capture the most relevant words contributing to sentiment classification.

5. **Model Training and Testing:**
   - The dataset is split into training (80%) and testing (20%) sets to evaluate the model’s generalization capability.
   - A **Logistic Regression** model is trained on the processed textual data.
   - The model is then used to predict sentiments on the test data.

6. **Model Evaluation:**
   - The **accuracy score** is calculated to measure overall performance.
   - A **classification report** is generated to analyze precision, recall, and F1-score.
   - A **confusion matrix** is plotted using Seaborn to visualize correct and incorrect predictions.

7. **Predicting Sentiments on New Reviews:**
   - A separate dataset containing new Amazon reviews is processed similarly.
   - The pre-trained logistic regression model is used to predict the sentiment of new reviews.
   - The results, including predicted sentiment and corresponding labels (Positive/Negative), are saved into a CSV file.

### Applications of Sentiment Analysis
Sentiment analysis is widely applicable across various industries, enabling businesses and organizations to gain valuable insights into customer opinions and market trends. The key applications of this project include:

1. **E-Commerce and Customer Feedback Analysis:**
   - Online retailers like Amazon, Flipkart, and eBay can analyze customer sentiments to understand their satisfaction levels.
   - Brands can improve their products by addressing issues highlighted in negative reviews.

2. **Brand Reputation Management:**
   - Companies can monitor customer feedback across different platforms to identify trends in customer satisfaction or dissatisfaction.
   - Real-time sentiment analysis allows brands to respond quickly to emerging customer concerns.

3. **Product Development and Enhancement:**
   - Businesses can gather insights into product performance by analyzing positive and negative sentiments.
   - Product teams can prioritize features and fixes based on customer feedback trends.

4. **Market Research and Competitor Analysis:**
   - Analyzing reviews for competing products helps businesses identify strengths and weaknesses in the market.
   - Businesses can use sentiment analysis to assess how their products compare to competitors.

5. **Social Media Monitoring:**
   - Companies can analyze social media comments, tweets, and reviews to understand public perception.
   - Marketing teams can tailor campaigns based on sentiment trends to increase engagement and positive brand associations.

6. **Customer Support Automation:**
   - Sentiment analysis can be integrated into chatbot systems to prioritize urgent customer complaints.
   - AI-driven customer support can provide instant responses based on sentiment detection.

### Conclusion
This project effectively demonstrates the implementation of sentiment analysis on Amazon product reviews using logistic regression. By leveraging NLP techniques and machine learning models, customer sentiments are accurately classified, enabling businesses to gain insights into customer feedback. The methodology used in this project can be extended to other domains such as finance, healthcare, and social media analysis, showcasing the versatility and impact of sentiment analysis in today's data-driven world.

