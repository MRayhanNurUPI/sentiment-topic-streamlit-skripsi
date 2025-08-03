import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import re

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

st.set_page_config(page_title="Dashboard Sentimen & Topik", layout="wide")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_data
def load_normalization_dict():
    df_norm = pd.read_csv("colloquial-indonesian-lexicon.csv")
    return dict(zip(df_norm["slang"], df_norm["formal"]))

@st.cache_data
def load_stopwords():
    return set(pd.read_csv("stopwordbahasa.csv", header=None)[0].tolist())

@st.cache_resource
def load_relevansi_model():
    model_name = "mrayhannurUPI/finetuned_indobert_praproses_relevansi_xiaomi"
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    return tokenizer, model

@st.cache_resource
def load_sentiment_model():
    model_name = "mrayhannurUPI/finetuned_indobert_sentiment_xiaomi"
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    class_names = ['negatif', 'netral', 'positif']
    return tokenizer, model, class_names

def clean_text(text, norm_dict, stopwords):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [norm_dict.get(word, word) for word in words if word not in stopwords]
    return ' '.join(words)

def predict_relevansi(texts, tokenizer, model, batch_size=16):
    labels = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_clean = [t for t in batch_texts if "xiaomi" in t.lower() or "xiaomiindonesia" in t.lower()]
        labels.extend([0 if t not in batch_clean else 1 for t in batch_texts])
    return labels

def predict_sentiment_batch(texts, tokenizer, model, class_names, batch_size=32):
    sentiments = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts.tolist(), return_tensors='pt', padding=True, truncation=True, max_length=256)
        with torch.inference_mode():
            preds = torch.argmax(model(**inputs).logits, dim=1)
        sentiments.extend([class_names[p] for p in preds])
    return sentiments

@st.cache_resource
def run_bertopic_model(texts):
    topic_model = BERTopic(
        embedding_model="all-MiniLM-L6-v2",
        umap_model=UMAP(n_neighbors=5, n_components=10, min_dist=0.1),
        hdbscan_model=HDBSCAN(min_cluster_size=max(2, int(len(texts)*0.006))),
        vectorizer_model=CountVectorizer(stop_words=["ya", "gue", "banget"]),
        language="multilanguage"
    )
    topics, _ = topic_model.fit_transform(texts)
    return topic_model, topics

def main():
    st.title("\U0001F4CA Dashboard Analisis Sentimen & Topik Xiaomi")
    uploaded_file = st.file_uploader("Unggah file CSV dengan kolom 'timestamp' & 'text'", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if set(df.columns) >= {"timestamp", "text"}:
            norm_dict = load_normalization_dict()
            stopwords = load_stopwords()
            df["clean_text"] = df["text"].apply(lambda x: clean_text(x, norm_dict, stopwords))

            rel_tokenizer, rel_model = load_relevansi_model()
            df["relevan"] = predict_relevansi(df["clean_text"], rel_tokenizer, rel_model)
            df = df[df["relevan"] == 1]

            sent_tokenizer, sent_model, sent_names = load_sentiment_model()
            df["sentiment"] = predict_sentiment_batch(df["clean_text"], sent_tokenizer, sent_model, sent_names)

            topic_model, topics = run_bertopic_model(df["clean_text"].tolist())
            df["topic_id"] = topics
            topic_info = topic_model.get_topic_info()
            topic_dict = topic_model.get_topics()

            keyword_map = {
                topic: ", ".join([w for w, _ in topic_dict[topic][:8]])
                for topic in topic_info.Topic
            }
            df["keywords"] = df["topic_id"].map(keyword_map)

            col1, col2 = st.columns(2)
            with col1:
                selected_sentiment = st.selectbox("Filter Sentimen", options=["All"] + sent_names)
            with col2:
                topic_options = df["topic_id"].unique().tolist()
                selected_topic = st.selectbox("Filter Topik", options=["All"] + topic_options)

            filtered_df = df.copy()
            if selected_sentiment != "All":
                filtered_df = filtered_df[filtered_df["sentiment"] == selected_sentiment]
            if selected_topic != "All":
                filtered_df = filtered_df[filtered_df["topic_id"] == selected_topic]

            st.dataframe(filtered_df[["timestamp", "text", "sentiment", "topic_id", "keywords"]])

            with st.expander("Lihat Ringkasan Topik"):
                summary_df = topic_info.rename(columns={"Topic": "topic_id", "Count": "docs_count"})
                summary_df["keywords"] = summary_df["topic_id"].map(keyword_map)
                st.dataframe(summary_df[["topic_id", "docs_count", "keywords"]])
        else:
            st.error("Kolom 'timestamp' dan 'text' tidak ditemukan.")

if __name__ == "__main__":
    main()



