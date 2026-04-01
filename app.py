import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

st.set_page_config(page_title="Opinion Mining Hub", page_icon="💬", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button { border-radius: 20px; width: 100%; transition: 0.3s; }
    .stButton>button:hover { background-color: #4f46e5; color: white; transform: scale(1.02); }
    .stMetric { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; }
    div[data-testid="stExpander"] { border-radius: 10px; border: 1px solid #e2e8f0; }
    .stProgress > div > div > div > div { background-color: #4f46e5; }
</style>
""", unsafe_allow_html=True)

# Seaborn/Matplotlib global style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.family'] = 'sans-serif'

# NLTK dependencies download
@st.cache_resource
def download_nltk_deps():
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')

download_nltk_deps()

# Initialize NLTK tools globally for efficiency
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

st.title("🗣️ Sentiment Analysis and Opinion Mining Hub")
st.markdown("Analyze social media posts, comments, and reviews to understand public sentiment towards brands, products, events, or topics.")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Tweets.csv")
        return pd.DataFrame(df)
    except FileNotFoundError:
        st.error("Dataset 'Tweets.csv' not found. Please wait while it generates or run generate_data.py")
        return pd.DataFrame()

df = load_data()

st.sidebar.title("Pipeline Settings")
analysis_option = st.sidebar.radio("Choose Analysis Type", ["Dataset Exploration", "Predict Custom Text", "Opinion Mining & Visualizations"])

if df.empty:
    st.warning("Failed to load dataset.")
    st.stop()

# Helper function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'@[A-Za-z0-9_]+', '', text) # Remove mentions
    text = re.sub(r'#', '', text) # Remove hashtags
    text = re.sub(r'RT[\s]+', '', text) # Remove retweets
    text = re.sub(r'https?:\/\/\S+', '', text) # Remove links
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove special characters
    text = text.lower() # lowercase
    words = text.split()
    words = [LEMMATIZER.lemmatize(word) for word in words if word not in STOP_WORDS]
    return ' '.join(words)

# Preprocessing cache
@st.cache_data
def get_clean_data(dataframe):
    dataframe['clean_text'] = dataframe['text'].apply(preprocess_text)
    return dataframe

with st.spinner("Processing text data..."):
    clean_df = get_clean_data(df)

# Model Training cache
@st.cache_resource
def train_model(dataframe):
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(dataframe['clean_text'])
    y = dataframe['airline_sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, vectorizer, accuracy

with st.spinner("Training model..."):
    model, vectorizer, accuracy = train_model(clean_df)

if analysis_option == "Dataset Exploration":
    st.header("Dataset Overview")
    st.markdown("We are using the **Twitter US Airline Sentiment** dataset which contains tweets directed at major US airlines.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Reviews (Tweets)", len(clean_df))
    with col2:
        st.metric("Model Accuracy (Logistic Regression)", f"{accuracy*100:.2f}%")
        
    st.subheader("Raw Data Sample")
    st.dataframe(df[['airline', 'airline_sentiment', 'text']].head(10), use_container_width=True)
    
    st.subheader("Cleaned Data Sample")
    st.dataframe(clean_df[['airline_sentiment', 'clean_text']].head(10), use_container_width=True)

elif analysis_option == "Predict Custom Text":
    st.header("Live Sentiment Analyzer")
    st.markdown("Enter a custom review or tweet to analyze its sentiment.")
    
    user_input = st.text_area("Enter Text:", "I had a wonderful flight experience today! The crew was so polite.")
    
    if st.button("Analyze Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            with st.spinner("Analyzing..."):
                cleaned_input = preprocess_text(user_input)
                vectorized_input = vectorizer.transform([cleaned_input])
                prediction = model.predict(vectorized_input)[0]
                probabilities = model.predict_proba(vectorized_input)[0]
                
                classes = model.classes_
                prob_dict = dict(zip(classes, probabilities))
                
                sentiment_colors = {
                    "positive": "green",
                    "neutral": "gray",
                    "negative": "red"
                }
                
                st.markdown(f"### Predicted Sentiment: **<span style='color:{sentiment_colors[prediction]}'>{prediction.capitalize()}</span>**", unsafe_allow_html=True)
                
                st.subheader("Confidence Scores")
                cols = st.columns(len(prob_dict))
                for idx, (cls, prob) in enumerate(prob_dict.items()):
                    with cols[idx]:
                        st.metric(label=cls.capitalize(), value=f"{prob*100:.1f}%")
                        st.progress(prob)

elif analysis_option == "Opinion Mining & Visualizations":
    st.header("Opinion Mining Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Sentiment Distribution", "Word Clouds", "Brand Sentiment (Airlines)"])
    
    with tab1:
        st.subheader("Overall Sentiment Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=clean_df, x='airline_sentiment', palette="Set2", ax=ax, order=["negative", "neutral", "positive"])
        ax.set_title("Count of Sentiments")
        ax.set_ylabel("Count")
        ax.set_xlabel("Sentiment")
        st.pyplot(fig)
        
    with tab2:
        st.subheader("Word Clouds by Sentiment")
        sentiment_type = st.radio("Select Sentiment to View Word Cloud", ["negative", "neutral", "positive"])
        
        words = ' '.join([text for text in clean_df[clean_df['airline_sentiment'] == sentiment_type]['clean_text']])
        if len(words) > 0:
            wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110, background_color="white").generate(words)
            
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.write("No words found for this category.")
            
    with tab3:
        st.subheader("Sentiment Distribution per Airline (Opinion Mining)")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(data=clean_df, x='airline', hue='airline_sentiment', palette="Set2", ax=ax)
        ax.set_title("Sentiment Breakdown by Airline")
        ax.set_ylabel("Number of Tweets")
        ax.set_xlabel("Airline")
        plt.legend(title='Sentiment')
        st.pyplot(fig)
