# Sentiment_Analysis_Webapp
# 🎭 Sentiment Analysis Pro

A Streamlit web application that performs end-to-end sentiment analysis on text data. Upload a CSV of reviews, and the app will automatically preprocess the data, train multiple machine learning models (Logistic Regression, SVM, etc.), and select the best one to analyze the sentiment.

### live demo link: 

---

## ✨ Features

* **📊 CSV Upload:** Analyze your own datasets (e.g., customer reviews, feedback).
* **🤖 Automated ML:** Automatically trains Logistic Regression, Multinomial Naive Bayes, Linear SVM, and Random Forest models.
* **🏆 Best Model Selection:** Automatically selects the best-performing model based on F1-score for predictions.
* **📈 Interactive Visualizations:**
    * Sentiment distribution pie charts.
    * Model performance comparison bar chart.
    * Confusion matrix and classification report.
    * Word clouds for Positive, Negative, and Neutral sentiments.
* **🌐 Multi-Language Support:** Automatically detects and translates text from any language to English for analysis.
* **💬 Real-Time Prediction:** Test the trained model with your own custom text inputs.
* **💾 Export Results:** Download the full analysis (with predictions) as a CSV or Excel file.

---

## 🚀 How to Use

1.  **Upload & Analyze:**
    * Navigate to the "📁 Upload & Analyze" tab.
    * Upload a CSV file containing columns for text (e.g., 'review_text') and ratings (e.g., 'rating').
    * Select the correct text and rating columns.
    * Click "🚀 Start Sentiment Analysis".
    * Review the model performance and visualizations.
2.  **Real-Time Prediction:**
    * Navigate to the "💬 Real-time Prediction" tab.
    * Type or paste any text into the text area.
    * Click "🔮 Analyze Sentiment" to get an instant prediction.

---

## 🛠️ Technologies Used

* **Python:** The core programming language.
* **Streamlit:** For building and deploying the web app.
* **Scikit-learn:** For machine learning models (Logistic Regression, SVM, RF, MNB) and TF-IDF vectorization.
* **Pandas:** For data manipulation.
* **NLTK:** For text preprocessing (stopwords, lemmatization).
* **Plotly & Seaborn:** For interactive charts and data visualization.
* **WordCloud:** For generating word clouds.
* **deep-translator & langdetect:** For multi-language translation and detection.
