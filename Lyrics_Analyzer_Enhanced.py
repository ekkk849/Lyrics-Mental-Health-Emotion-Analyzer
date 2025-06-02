#!/usr/bin/env python
# coding: utf-8

# # 🎶 Enhanced Lyrics Mental Health & Emotion Analyzer
# Now with visualizations and file upload support!

# In[ ]:


import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[ ]:


vectorizer = joblib.load("vectorizer.pkl")
emotion_model = joblib.load("emotion_model.pkl")


# In[ ]:


def analyze_depression(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']

    # Display VADER score
    st.write("### VADER Sentiment Breakdown")
    fig, ax = plt.subplots()
    labels = ['Positive', 'Negative', 'Neutral']
    values = [scores['pos'], scores['neg'], scores['neu']]
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    st.pyplot(fig)

    # Mood classification based on compound
    st.write(f"VADER Compound Score: `{compound}`")
    if compound <= -0.5:
        st.error("⚠️ Potential signs of depression.")
    elif compound < 0:
        st.warning("🟠 Slightly negative tone.")
    else:
        st.success("🟢 Positive or neutral tone.")


# In[ ]:


def predict_emotion(text, model, vectorizer):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]

    st.write("### Emotion Classification Result")
    st.info(f"Predicted Emotion: **{pred.upper()}**")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        classes = model.classes_
        proba_df = pd.DataFrame({'Emotion': classes, 'Probability': proba})

        fig, ax = plt.subplots()
        sns.barplot(data=proba_df, x='Emotion', y='Probability', palette='pastel', ax=ax)
        ax.set_ylim(0, 1)
        ax.set_title("Emotion Probabilities")
        st.pyplot(fig)


# In[ ]:


st.set_page_config(page_title="Lyrics Analyzer", layout="centered")
st.title("🎶 Lyrics Mental Health & Emotion Analyzer")

input_type = st.radio("Select Input Method", ["Manual Entry", "Upload .txt File"])

if input_type == "Manual Entry":
    user_input = st.text_area("Enter your lyrics below:", height=200)
elif input_type == "Upload .txt File":
    uploaded_file = st.file_uploader("Upload a .txt file with lyrics", type="txt")
    if uploaded_file:
        user_input = uploaded_file.read().decode("utf-8")
    else:
        user_input = ""

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter or upload some lyrics.")
    else:
        st.subheader("🔍 Depression Analysis (VADER)")
        analyze_depression(user_input)

        st.subheader("🎭 Emotion Classification")
        predict_emotion(user_input, emotion_model, vectorizer)

