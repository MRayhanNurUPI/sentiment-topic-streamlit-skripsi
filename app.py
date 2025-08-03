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

# Load sentiment model
@st.cache_resource
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    model = AutoModelForSequenceClassification.from_pretrained("rayhansaputra/indobert-sentiment-topic-skripsi")
    class_names = ['Negatif', 'Netral', 'Positif']
    return tokenizer, model, class_names

# Load topic model
@st.cache_resource
def run_bertopic_model(texts):
    vectorizer_model = CountVectorizer(stop_words="english")
    topic_model = BERTopic(vectorizer_model=vectorizer_model, calculate_probabilities=False)
    topics, _ = topic_model.fit_transform(texts)
    return topic_model, topics

# Preprocess text
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    return text

# Predict sentiment in batches
def predict_sentiment_batch(texts, batch_size=32):
    sentiments = []
    total = len(texts)
    progress_text = "⏳ Memproses prediksi sentimen..."
    progress_bar = st.progress(0, text=progress_text)

    for i in range(0, total, batch_size):
        batch_texts = texts[i:i + batch_size].tolist()
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=256)
        with torch.inference_mode():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            sentiments.extend([class_names[p] for p in preds])
        progress_bar.progress(min((i + batch_size) / total, 1.0), text=progress_text)

    return sentiments


def plot_sentiment_per_topic(df, topic_model):
    st.subheader("\U0001F4CA Distribusi Sentimen per Topik (Top 10)")
    sentiment_colors = {
        2: '#66bb6a',
        1: '#b0bec5',
        0: '#ef5350'
    }

    df = df[df['topic_id'] != -1]
    top_topics = df['topic_id'].value_counts().nlargest(10).index.tolist()
    df_top = df[df['topic_id'].isin(top_topics)]

    df_top['label'] = df_top['sentiment'].map({'negatif': 0, 'netral': 1, 'positif': 2})
    sentiment_counts = df_top.groupby(['topic_id', 'label']).size().unstack(fill_value=0)
    sentiment_props = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0)

    def get_top_words_str(topic_num, topics_dict, n=8):
        if topic_num not in topics_dict:
            return "(no topic)"
        words = [word for word, _ in topics_dict[topic_num][:n]]
        return ' '.join(words)

    topics_dict = topic_model.get_topics()
    y_labels = [f"Topik {cluster}:\n{get_top_words_str(cluster, topics_dict)}" for cluster in top_topics]

    fig, ax = plt.subplots(figsize=(12, len(top_topics) * 0.7))
    bottom = np.zeros(len(top_topics))

    for sentiment in [0, 1, 2]:
        values = sentiment_props.get(sentiment, pd.Series([0]*len(top_topics)))
        bars = ax.barh(range(len(top_topics)), values, left=bottom, color=sentiment_colors[sentiment], label=sentiment)
        for j, (val, left) in enumerate(zip(values, bottom)):
            if val > 0.03:
                ax.text(left + val / 2, j, f"{val*100:.1f}%", ha='center', va='center', fontsize=9)
        bottom += values

    ax.set_yticks(range(len(top_topics)))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Proporsi Sentimen", fontsize=12)
    ax.set_title("Distribusi Sentimen dan Kata Kunci per Topik", fontsize=14, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    st.pyplot(fig)

# Main Streamlit app
def main():
    st.title("📊 Analisis Sentimen dan Topik Tweet")
    st.write("Unggah file CSV tweet untuk dianalisis.")

    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding='latin1')

        st.success("✅ File berhasil diunggah.")
        if "clean_text" not in df.columns or "relevan" not in df.columns:
            st.error("❌ Kolom 'clean_text' dan 'relevan' wajib ada.")
            return

        df = df[df["relevan"] == 1]

        # Load models
        sent_tokenizer, sent_model, class_names = load_sentiment_model()
        global tokenizer, model
        tokenizer, model = sent_tokenizer, sent_model

        # Predict sentiment
        df["sentiment"] = predict_sentiment_batch(df["clean_text"])

        # Run topic modeling with progress bar
        st.write("⏳ Memproses pemodelan topik...")
        topic_progress = st.empty()
        topic_progress.progress(0.5, text="Sedang memproses model topik...")

        topic_model, topics = run_bertopic_model(df["clean_text"])
        df["topic_id"] = topics

        topic_progress.progress(1.0, text="✅ Pemodelan topik selesai.")

        st.subheader("🔍 Filter Data")

        # Filter options
        selected_topics = st.multiselect(
            "Pilih Cluster Topik (Topic ID)", options=sorted(df["topic_id"].unique()),
            default=sorted(df["topic_id"].unique())
        )

        selected_sentiments = st.multiselect(
            "Pilih Label Sentimen", options=class_names, default=class_names
        )

        # Apply filter
        filtered_df = df[
            (df["topic_id"].isin(selected_topics)) &
            (df["sentiment"].isin(selected_sentiments))
        ]

        st.write("📄 Dataframe hasil setelah filter:")
        st.dataframe(filtered_df)

        # Plot sentiment
        st.subheader("📈 Distribusi Sentimen (Data Terfilter)")
        fig, ax = plt.subplots()
        sns.countplot(x="sentiment", data=filtered_df, order=class_names, palette="pastel", ax=ax)
        st.pyplot(fig)
        
        # Plot topic
        st.subheader("📊 Distribusi Topik (Data Terfilter)")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        filtered_df["topic_id"].value_counts().sort_index().plot(kind='bar', color="skyblue", ax=ax2)
        ax2.set_xlabel("Topic ID")
        ax2.set_ylabel("Jumlah")
        st.pyplot(fig2)

        # Visualisasi Distribusi Sentimen per Topik
        plot_sentiment_per_topic(df, topic_model)
        
        # Download button
        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Unduh Data Terfilter", data=csv, file_name="hasil_filter.csv", mime="text/csv")

if __name__ == "__main__":
    main()
