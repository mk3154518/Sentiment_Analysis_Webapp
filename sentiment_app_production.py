import streamlit as st
import pandas as pd
import numpy as np
import re
import time
import joblib
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px
import plotly.graph_objects as go

# Import ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)


# Import NLP libraries
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer

# Language detection and translation
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator

# Set random seed for consistent language detection
DetectorFactory.seed = 0

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Pro",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Download NLTK data
@st.cache_resource
def initialize_nltk():
    """Download required NLTK data"""
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        stop_words = set(nltk_stopwords.words('english'))
        # Keep negative words
        stop_words -= {'not', 'no', 'never', 'nor', 'neither'}
        return stop_words
    except Exception as e:
        st.warning(f"NLTK initialization warning: {e}")
        return set(STOPWORDS)

# Initialize
STOP_WORDS = initialize_nltk()
lemmatizer = WordNetLemmatizer()

# =========================
# PREPROCESSING FUNCTIONS
# =========================

@st.cache_data(show_spinner=False)
def detect_language_safe(text):
    """Detect language with error handling"""
    try:
        return detect(str(text)[:1000])  # Limit text length
    except:
        return 'en'

@st.cache_data(show_spinner=False)
def translate_text(text, source='auto', target='en'):
    """Translate text to target language"""
    try:
        if len(str(text).strip()) < 3:
            return text

        detected = detect(str(text)[:1000])
        if detected == target:
            return text

        translator = GoogleTranslator(source=source, target=target)
        return translator.translate(str(text)[:5000])  # Limit to 5000 chars
    except Exception as e:
        return text

def clean_text_data(text):
    """Clean and normalize text"""
    if not isinstance(text, str) or not text.strip():
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Keep only letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

@st.cache_data(show_spinner=False)
def tokenize_and_lemmatize(text):
    """Tokenize, remove stopwords, and lemmatize"""
    if not text:
        return ""

    words = text.split()
    processed = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in STOP_WORDS and len(word) > 2
    ]

    return ' '.join(processed)

def create_sentiment_label(rating, neg_thresh, neu_val):
    """Map rating to sentiment using dynamic thresholds"""
    try:
        r = int(float(rating))
        if r <= neg_thresh:
            return 'Negative'
        elif r == neu_val:
            return 'Neutral'
        elif r > neu_val: # This ensures Positive is anything greater than Neutral
            return 'Positive'
        else:
            # Handle cases where rating is between neg_thresh and neu_val if they are non-adjacent
            # We'll treat this as Neutral or the lowest positive rating
            return 'Neutral' 
    except:
        return 'Neutral'

# =========================
# MODEL TRAINING
# =========================

def train_multiple_models(X_train, y_train, X_test, y_test):
    """Train and evaluate multiple models"""

    models_to_train = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            solver='liblinear',
            random_state=42,
            class_weight='balanced'
        ),
        'Multinomial NB': MultinomialNB(alpha=1.0),
        'Linear SVM': LinearSVC(
            max_iter=5000,
            random_state=42,
            dual=False,
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            max_depth=20
        )
    }

    results = {}
    progress_container = st.container()

    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, (name, model) in enumerate(models_to_train.items()):
            status_text.markdown(f"**Training {name}...**")

            # Train
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time

            # Predict
            y_pred = model.predict(X_test)

            # Calculate metrics
            results[name] = {
                'model': model,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'predictions': y_pred,
                'train_time': train_time
            }

            progress_bar.progress((idx + 1) / len(models_to_train))

        progress_bar.empty()
        status_text.empty()

    # Find best model by F1 score
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])

    return results, best_model_name

# =========================
# VISUALIZATION FUNCTIONS
# =========================

def plot_sentiment_distribution(sentiment_series):
    """Create pie chart for sentiment distribution"""
    counts = sentiment_series.value_counts()

    colors = {
        'Positive': '#2ECC71',
        'Negative': '#E74C3C',
        'Neutral': '#F39C12'
    }

    fig = go.Figure(data=[go.Pie(
        labels=counts.index,
        values=counts.values,
        marker=dict(colors=[colors.get(label, '#95A5A6') for label in counts.index]),
        hole=0.4,
        textinfo='label+percent+value',
        textfont=dict(size=14)
    )])

    fig.update_layout(
        title={
            'text': 'Sentiment Distribution',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=450,
        showlegend=True
    )

    return fig

def plot_model_metrics(results_df):
    """Create comparison chart for model metrics"""

    fig = go.Figure()

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    colors = ['#3498DB', '#2ECC71', '#F39C12', '#E74C3C']

    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(
            name=metric,
            x=results_df['Model'],
            y=results_df[metric],
            marker_color=color,
            text=results_df[metric].round(3),
            textposition='auto',
        ))

    fig.update_layout(
        title='Model Performance Metrics',
        xaxis_title='Model',
        yaxis_title='Score',
        barmode='group',
        height=500,
        yaxis=dict(range=[0, 1.05]),
        hovermode='x unified'
    )

    return fig

def generate_wordcloud_for_sentiment(df, sentiment, title):
    """Generate word cloud for specific sentiment"""

    text_data = ' '.join(df[df['Sentiment'] == sentiment]['processed'].values)

    if not text_data.strip():
        return None

    # Custom stopwords
    custom_stopwords = set(STOPWORDS)
    custom_stopwords.update(['phone', 'mobile', 'product', 'camera', 'battery'])

    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color='white',
        colormap='viridis',
        max_words=150,
        stopwords=custom_stopwords,
        relative_scaling=0.5,
        min_font_size=10,
        collocations=False
    ).generate(text_data)

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout(pad=0)

    return fig

def plot_confusion_matrix(cm, labels, title):
    """Create confusion matrix heatmap"""

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='RdYlGn',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={'label': 'Count'},
        linewidths=1,
        linecolor='gray'
    )

    ax.set_ylabel('Actual Sentiment', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Sentiment', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)

    plt.tight_layout()

    return fig

# =========================
# MAIN APPLICATION
# =========================

def main():

    # Header
    st.markdown('<h1 class="main-header">🎭 Sentiment Analysis Pro</h1>', 
                unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center; padding: 1rem; background-color: #F0F2F6; border-radius: 10px; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; margin: 0;'>
            📊 Upload CSV → 🌐 Translate → 🧹 Clean → 🤖 Train Models → 📈 Visualize → 💾 Download
        </p>
    </div>
    """, unsafe_allow_html=True)
    models = load_models()
    if models[0] is not None:
        store_models(models[0], models[1])

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        st.markdown("### Model Configuration")
        max_features = st.slider("Max TF-IDF Features", 1000, 10000, 5000, 500)
        test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5) / 100

        st.markdown("### Sentiment Thresholds")

# Negative (e.g., 1-2)
        negative_threshold = st.number_input(
            "Negative Rating (≤)", 1, 3, 2, 
            help="Ratings less than or equal to this are classified as Negative."
        )

# Neutral (must be > negative_threshold)
# Setting the min to negative_threshold + 1 and default to 3
        default_neutral = min(5, max(negative_threshold + 1, 3))

        neutral_value = st.number_input(
            "Neutral Rating (=)",
            min_value=negative_threshold + 1,
            max_value=5,
            value=default_neutral,
            help="Rating equal to this is classified as Neutral."
        )
        st.markdown("---")
        st.markdown("### 💾 Save Trained Model")
        
        # Check if an analysis has been run
        if 'analysis_done' in st.session_state:
            st.info("A model is trained and ready to save.")
            
            if st.button("💾 Save Model Files", use_container_width=True):
                try:
                    # Read all data from session state
                    best_model = st.session_state['best_model']
                    vectorizer = st.session_state['vectorizer']
                    metadata = st.session_state['model_metadata']

                    # Save the files
                    joblib.dump(best_model, 'best_sentiment_model.joblib')
                    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
                    joblib.dump(metadata, 'model_metadata.joblib')
                    
                    st.success("✅ Models saved!")
                    st.caption("Files saved: `best_sentiment_model.joblib`, `tfidf_vectorizer.joblib`, `model_metadata.joblib`")
                
                except Exception as e:
                    st.error(f"Error saving model: {e}")
        else:
            st.info("Run an analysis first to enable saving.")
        st.markdown("### 📚 About")
        st.info("""
        This app performs sentiment analysis on text data:

        ✅ Multi-language support
        ✅ Automatic model selection
        ✅ Real-time predictions
        ✅ Interactive visualizations
        ✅ Export results
        """)

    # Main content
    tab1, tab2 = st.tabs(["📁 Upload & Analyze", "💬 Real-time Prediction"])

    # Tab 1: File Upload and Analysis
    with tab1:
        st.header("Step 1: Upload Your Dataset")

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with text data and ratings"
        )

        if uploaded_file is not None:

            try:
                # Read file
                df = pd.read_csv(uploaded_file)

                st.success(f"✅ File uploaded: {df.shape[0]} rows × {df.shape[1]} columns")

                # Preview
                with st.expander("👀 Preview Dataset", expanded=False):
                    st.dataframe(df.head(10), use_container_width=True)
                    st.markdown(f"**Columns:** {', '.join(df.columns.tolist())}")

                # Column selection
                st.header("Step 2: Configure Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    # Text column
                    text_columns = df.select_dtypes(include=['object']).columns.tolist()

                    if not text_columns:
                        st.error("❌ No text columns found!")
                        return

                    # Check for title + body pattern
                    has_title_body = 'title' in df.columns and 'body' in df.columns

                    if has_title_body:
                        use_combined = st.checkbox("Combine 'title' + 'body' columns", value=True)
                        if use_combined:
                            text_column = 'combined_text'
                            df['combined_text'] = df['title'].fillna('') + ' ' + df['body'].fillna('')
                        else:
                            text_column = st.selectbox("Select text column:", text_columns)
                    else:
                        text_column = st.selectbox("Select text column:", text_columns)

                with col2:
                    # Rating column
                    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

                    if numeric_columns:
                        rating_column = st.selectbox("Select rating column:", numeric_columns)
                    else:
                        st.error("❌ No numeric rating column found!")
                        return

                # Analysis button
                st.header("Step 3: Run Analysis")

                analyze_button = st.button(
                    "🚀 Start Sentiment Analysis",
                    type="primary",
                    use_container_width=True
                )

                if analyze_button:

                    # Create analysis container
                    with st.container():

                        # ============ PREPROCESSING ============
                        st.markdown("### 📊 Phase 1: Data Preprocessing")

                        progress_bar = st.progress(0, text="Starting...")

                        # Step 1: Language Detection
                        progress_bar.progress(10, text="🔍 Detecting languages...")
                        df['language'] = df[text_column].apply(detect_language_safe)

                        lang_counts = df['language'].value_counts()
                        st.info(f"Languages detected: {', '.join([f'{lang} ({count})' for lang, count in lang_counts.items()])}")

                        # Step 2: Translation
                        progress_bar.progress(25, text="🌐 Translating to English...")
                        df['translated'] = df[text_column].apply(translate_text)

                        # Step 3: Cleaning
                        progress_bar.progress(40, text="🧹 Cleaning text...")
                        df['cleaned'] = df['translated'].apply(clean_text_data)

                        # Step 4: Tokenization
                        progress_bar.progress(55, text="✂️ Tokenizing & Lemmatizing...")
                        df['processed'] = df['cleaned'].apply(tokenize_and_lemmatize)

                        # Step 5: Remove empty
                        df = df[df['processed'].str.len() > 10].reset_index(drop=True)

                        progress_bar.progress(70, text="✅ Preprocessing complete!")
                        time.sleep(0.3)
                        progress_bar.empty()

                        st.success(f"✅ Preprocessed {len(df)} valid records")

                        # Create sentiment labels
                        if rating_column:
                            df['Sentiment'] = df[rating_column].apply(
                                lambda x: create_sentiment_label(x, negative_threshold, neutral_value)
                            )

                       # Display stats
                        col1, col2, col3, col4 = st.columns(4)
                        sentiment_counts = df['Sentiment'].value_counts() # Calculate once

                        with col1:
                            st.metric("📝 Total Records", len(df))
                        with col2:
                            st.metric("😊 Positive", sentiment_counts.get('Positive', 0))
                        with col3:
                            st.metric("😐 Neutral", sentiment_counts.get('Neutral', 0)) 
                        with col4:
                            st.metric("😞 Negative", sentiment_counts.get('Negative', 0))

                        # Show preprocessing sample
                        with st.expander("🔍 View Preprocessing Example"):
                            sample_idx = 0
                            st.markdown(f"**Original:** {df[text_column].iloc[sample_idx][:200]}...")
                            st.markdown(f"**Translated:** {df['translated'].iloc[sample_idx][:200]}...")
                            st.markdown(f"**Cleaned:** {df['cleaned'].iloc[sample_idx][:200]}...")
                            st.markdown(f"**Processed:** {df['processed'].iloc[sample_idx][:200]}...")

                        # ============ MODEL TRAINING ============
                        st.markdown("### 🤖 Phase 2: Model Training & Evaluation")

                        # Vectorization
                        with st.spinner("Creating TF-IDF features..."):
                            vectorizer = TfidfVectorizer(
                                max_features=max_features,
                                ngram_range=(1, 2),
                                min_df=2,
                                max_df=0.95
                            )

                            X = vectorizer.fit_transform(df['processed'])
                            y = df['Sentiment']

                            st.info(f"Feature matrix shape: {X.shape}")

                        # Train-test split
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y,
                            test_size=test_size,
                            random_state=42,
                            stratify=y
                        )

                        st.info(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

                        # Train models
                        results, best_model_name = train_multiple_models(X_train, y_train, X_test, y_test)

                        # Create results dataframe
                        results_data = {
                            'Model': list(results.keys()),
                            'Accuracy': [results[m]['accuracy'] for m in results.keys()],
                            'Precision': [results[m]['precision'] for m in results.keys()],
                            'Recall': [results[m]['recall'] for m in results.keys()],
                            'F1 Score': [results[m]['f1_score'] for m in results.keys()],
                            'Train Time (s)': [results[m]['train_time'] for m in results.keys()]
                        }

                        results_df = pd.DataFrame(results_data).sort_values('F1 Score', ascending=False)

                        # Display results
                        st.markdown("#### 📊 Model Comparison")

                        col1, col2 = st.columns([1, 2])

                        with col1:
                            st.dataframe(
                                results_df.style.background_gradient(subset=['F1 Score'], cmap='Greens')
                                                .format({
                                                    'Accuracy': '{:.4f}',
                                                    'Precision': '{:.4f}',
                                                    'Recall': '{:.4f}',
                                                    'F1 Score': '{:.4f}',
                                                    'Train Time (s)': '{:.2f}'
                                                }),
                                use_container_width=True
                            )

                        with col2:
                            fig = plot_model_metrics(results_df)
                            st.plotly_chart(fig, use_container_width=True)

                        # Highlight best model
                        best_f1 = results[best_model_name]['f1_score']
                        best_acc = results[best_model_name]['accuracy']

                        st.success(f"""
                        🏆 **Best Model:** {best_model_name}
                        - F1 Score: {best_f1:.4f}
                        - Accuracy: {best_acc:.4f}
                        """)
                        # === FIX: STORE MODELS IN SESSION STATE ===
                        best_model = results[best_model_name]['model']
                        store_models(vectorizer, best_model) # <--- Call the function here!
                        # ==========================================
                        st.session_state['model_metadata'] = {
                            'model_name': best_model_name,
                            'f1_score': best_f1,
                            'accuracy': best_acc,
                            'max_features': max_features
                        }
                        st.session_state['analysis_done'] = True
                        # ============ PREDICTIONS ============
                        st.markdown("### 🎯 Phase 3: Making Predictions")

                        best_model = results[best_model_name]['model']

                        with st.spinner("Predicting sentiments..."):
                            X_full = vectorizer.transform(df['processed'])
                            df['Predicted_Sentiment'] = best_model.predict(X_full)

                        st.success("✅ Predictions complete!")

                        # Show sample predictions
                        with st.expander("📋 Sample Predictions"):
                            display_cols = [text_column, rating_column, 'Sentiment', 'Predicted_Sentiment']
                            st.dataframe(df[display_cols].head(20), use_container_width=True)

                        # ============ CONFUSION MATRIX ============
                        st.markdown("### 📊 Phase 4: Model Evaluation")

                        y_pred_best = results[best_model_name]['predictions']

                        # Confusion matrix
                        cm = confusion_matrix(y_test, y_pred_best, 
                                            labels=['Negative', 'Neutral', 'Positive'])

                        col1, col2 = st.columns([1, 1])

                        with col1:
                            fig_cm = plot_confusion_matrix(
                                cm,
                                ['Negative', 'Neutral', 'Positive'],
                                f'Confusion Matrix - {best_model_name}'
                            )
                            st.pyplot(fig_cm)

                        with col2:
                            st.markdown("#### Classification Report")
                            report = classification_report(
                                y_test, y_pred_best,
                                target_names=['Negative', 'Neutral', 'Positive'],
                                output_dict=True
                            )

                            report_df = pd.DataFrame(report).T.iloc[:-3, :3]
                            st.dataframe(
                                report_df.style.format('{:.3f}').background_gradient(cmap='RdYlGn'),
                                use_container_width=True
                            )

                        # ============ VISUALIZATIONS ============
                        st.markdown("### 📈 Phase 5: Visualizations")

                        # Sentiment distribution
                        col1, col2 = st.columns([1, 1])

                        with col1:
                            st.markdown("#### Sentiment Distribution")
                            fig_dist = plot_sentiment_distribution(df['Sentiment'])
                            st.plotly_chart(fig_dist, use_container_width=True)

                        with col2:
                            st.markdown("#### Prediction Distribution")
                            fig_pred_dist = plot_sentiment_distribution(df['Predicted_Sentiment'])
                            st.plotly_chart(fig_pred_dist, use_container_width=True)

                        # Word clouds
                        st.markdown("#### ☁️ Word Clouds by Sentiment")

                        wc_col1, wc_col2, wc_col3 = st.columns(3)

                        sentiments = ['Positive', 'Negative', 'Neutral']
                        cols = [wc_col1, wc_col2, wc_col3]

                        for sentiment, col in zip(sentiments, cols):
                            with col:
                                fig_wc = generate_wordcloud_for_sentiment(
                                    df, sentiment, f'{sentiment} Reviews'
                                )
                                if fig_wc:
                                    st.pyplot(fig_wc)
                                else:
                                    st.info(f"No {sentiment} reviews")

                        # ============ DOWNLOAD ============
                        st.markdown("### 💾 Phase 6: Download Results")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            # CSV download
                            output_df = df[[text_column, rating_column, 'Sentiment', 
                                          'Predicted_Sentiment', 'language']]

                            csv = output_df.to_csv(index=False).encode('utf-8')

                            st.download_button(
                                label="📄 Download CSV",
                                data=csv,
                                file_name='sentiment_analysis_results.csv',
                                mime='text/csv',
                                use_container_width=True
                            )

                        with col2:
                            # Excel download
                            buffer = BytesIO()
                            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                output_df.to_excel(writer, index=False, sheet_name='Results')

                            st.download_button(
                                label="📊 Download Excel",
                                data=buffer.getvalue(),
                                file_name='sentiment_analysis_results.xlsx',
                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                use_container_width=True
                            )

                        with col3:
                            # Save model
                            if st.button("💾 Save Model", use_container_width=True):
                                joblib.dump(best_model, 'best_sentiment_model.joblib')
                                joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

                                # Create metadata
                                metadata = {
                                    'model_name': best_model_name,
                                    'f1_score': best_f1,
                                    'accuracy': best_acc,
                                    'max_features': max_features
                                }
                                joblib.dump(metadata, 'model_metadata.joblib')

                                st.success("✅ Model saved!")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                with st.expander("See error details"):
                    st.exception(e)

        else:
            # Instructions
            st.info("""
            👆 **Upload a CSV file to get started!**

            ### Expected Format:

            Your CSV should contain:
            - **Text data**: Reviews, comments, or any text
            - **Rating**: Numeric rating (e.g., 1-5 stars)

            ### Example:
            | text | rating |
            |------|--------|
            | Great product! | 5 |
            | Terrible quality | 1 |
            | It's okay | 3 |
            """)

    # Tab 2: Real-time Prediction
    with tab2:
        st.header("💬 Real-time Sentence Analysis")

        st.markdown("""
        Type or paste any text below to get instant sentiment prediction.
        The app will:
        1. Detect the language
        2. Translate to English (if needed)
        3. Preprocess the text
        4. Predict sentiment
        """)

        # Check if model is trained
        if 'vectorizer' not in st.session_state or 'best_model' not in st.session_state:
            st.warning("⚠️ Please train a model first by uploading and analyzing a dataset in the 'Upload & Analyze' tab.")
        else:
            user_input = st.text_area(
                "Enter text to analyze:",
                placeholder="Type your review or comment here...",
                height=150
            )

            col1, col2, col3 = st.columns([1, 1, 1])

            with col2:
                predict_btn = st.button("🔮 Analyze Sentiment", type="primary", use_container_width=True)

            if predict_btn and user_input.strip():

                with st.spinner("Analyzing..."):
                    # Process input
                    translated = translate_text(user_input)
                    cleaned = clean_text_data(translated)
                    processed = tokenize_and_lemmatize(cleaned)

                    # Predict
                    X_input = st.session_state['vectorizer'].transform([processed])
                    prediction = st.session_state['best_model'].predict(X_input)[0]

                    # Display result
                    st.markdown("---")

                    col1, col2 = st.columns([1, 2])

                    with col1:
                        sentiment_icons = {
                            'Positive': '😊',
                            'Negative': '😞',
                            'Neutral': '😐'
                        }

                        sentiment_colors = {
                            'Positive': 'green',
                            'Negative': 'red',
                            'Neutral': 'orange'
                        }

                        st.markdown(f"### {sentiment_icons[prediction]} Sentiment:")
                        st.markdown(f"## :{sentiment_colors[prediction]}[{prediction}]")

                    with col2:
                        st.markdown("### Processing Steps:")

                        steps_df = pd.DataFrame({
                            'Step': ['Original', 'Translated', 'Cleaned', 'Processed'],
                            'Text': [
                                user_input[:80] + '...' if len(user_input) > 80 else user_input,
                                translated[:80] + '...' if len(translated) > 80 else translated,
                                cleaned[:80] + '...' if len(cleaned) > 80 else cleaned,
                                processed[:80] + '...' if len(processed) > 80 else processed
                            ]
                        })

                        st.dataframe(steps_df, use_container_width=True, hide_index=True)

# Store trained models in session state
def store_models(vectorizer, model):
    """Store trained models in session state"""
    st.session_state['vectorizer'] = vectorizer
    st.session_state['best_model'] = model
@st.cache_resource
def load_models():
    """Load pre-trained models from disk"""
    try:
        vectorizer = joblib.load('tfidf_vectorizer.joblib')
        model = joblib.load('best_sentiment_model.joblib')
        return vectorizer, model
    except FileNotFoundError:
        # Return None if files are not found
        return None, None

if __name__ == "__main__":
    main()
