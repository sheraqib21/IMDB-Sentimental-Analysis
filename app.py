import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


dataset = pd.read_csv('IMDBDataset.csv')


vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(dataset['review']).toarray()
dataset['class'] = dataset['class'].map({'positive': 1, 'negative': 0})
y = dataset['class']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = GaussianNB()
model.fit(X_train, y_train)


accuracy = accuracy_score(y_test, model.predict(X_test))


st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬", layout="centered")
st.title("🎬 IMDB Sentiment Analysis")
st.write("Enter a movie review below, and let the AI predict whether the sentiment is **positive** or **negative**.")
st.markdown(f"**Model Accuracy:** {accuracy * 100:.2f}%")


st.subheader("Type Your Review:")
user_input = st.text_input("Press 'Enter' to analyze sentiment", "")


if user_input.strip() != "":
    
    input_vector = vectorizer.transform([user_input]).toarray()
    prediction = model.predict(input_vector)
    sentiment = "Positive 😊" if prediction[0] == 1 else "Negative 😞"
    
    
    st.markdown("---")
    st.subheader("Prediction Result:")
    st.markdown(f"**Sentiment:** {sentiment}")
    st.markdown("---")
