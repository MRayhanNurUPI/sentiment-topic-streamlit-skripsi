import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import re
import plotly.express as px

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

# === CONFIGURATIONS ===
st.set_page_config(page_title="Analisis Sentimen dan Topik", layout="wide")

# === GPU SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === RESOURCE LOADING ===
@st.cache_data
def load_normalization_dict():
    df_norm = pd.read_csv("colloquial-indonesian-lexicon.csv")
    return dict(zip(df_norm["slang"], df_norm["formal"]))

@st.cache_data
def load_stopwords():
    return set(pd.read_csv("stopwordbahasa.csv", header=None)[0].tolist())

@st.cache_resource
def load_relevansi_model():
    model_name = "mrayhannurUPI/finetuned_indobert_praproses_relevansi_xiaomi"  # Replace with actual HF model path
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")  # Or custom tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    class_names = ['tidak relevan', 'relevan']
    return tokenizer, model


def load_sentiment_model():
    # Load tokenizer and model
    model_name = "mrayhannurUPI/finetuned_indobert_sentiment_xiaomi"
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    class_names = ['negatif', 'netral', 'positif']
    return tokenizer, model, class_names

# === RELEVANCIES FILTERING ===
def predict_relevansi(texts, tokenizer, model, batch_size=16):
    labels = []
    total = len(texts)
    progress_bar = st.progress(0)

    for i in range(0, total, batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_clean = []

        for text in batch_texts:
            if "xiaomi" not in text.lower() and "xiaomiindonesia" not in text.lower():
                labels.append(0)
            else:
                batch_clean.append(text)

        if batch_clean:
            inputs = tokenizer(batch_clean, return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.inference_mode():
                logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=1)
            labels.extend(preds.cpu().tolist())

        percent_complete = min((i + batch_size) / total, 1.0)
        progress_bar.progress(percent_complete)

    return labels

# === TEXT CLEANING ===
def clean_text(text, norm_dict, stopwords):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = text.split()
    words = [norm_dict.get(word, word) for word in words]
    words = [word for word in words if word not in stopwords]
    return ' '.join(words)

# === SENTIMENT PREDICTION ===
def predict_sentiment_batch(texts, tokenizer, model, class_names, batch_size=16):
    sentiments = []
    total = len(texts)
    progress_bar = st.progress(0)

    for i in range(0, total, batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=1)
        sentiments.extend([class_names[p] for p in preds])

        percent_complete = min((i + batch_size) / total, 1.0)
        progress_bar.progress(percent_complete)

    return sentiments

# === TOPIC MODELING ===
@st.cache_resource
def run_bertopic_model(texts):
    embedding_model = "all-MiniLM-L6-v2"
    umap_model = UMAP(n_neighbors=5, n_components=10, min_dist=0.1)
    hdbscan_model = HDBSCAN(min_cluster_size = max(2, int(len(texts) * 0.006)), prediction_data=True)

    custom_stopwords = [
    "ya", "gue", 'gwehh', "banget", "kalo", 'bro'
    "mending", "nya", "ku", "gua", "deh", "sih",
    "nih", "kayak", 'xiaomiindonesia', 'ganti', 'bingung', "tau",
    "pas", "orang", "tuh", "wkwk", 'pengin', 'wtb', 'wts', 'kali', 'dimana',
    'terjangkau', 'pasar', 'kayaknya', 'bagusan',
    'bikin', 'guys', 'kemarin', 'dibanding', 'apa', 'tawarin', 'mas', 'ha',
    'anjir', 'lihat', 'sayang', 'temanku', 'doang', 'jual', 'beli', 'anak', 'pls',
    'kau', 'janjikan', 'butuh', 'ngejelasin', 'putuskan', 'mulu', 'kena', 'nak',
    'kembaran', 'better', 'kah', 'gusy', 'pon', 'mencari', 'cari', 'ilah', 'berkelana',
    'au', 'ane', 'lads', 'haha', "dong", 'allah', 'mengomong', 'memakai', 'its', 'joke',
    'ish', 'bora', 'eh', 'min', 'dm', 'biar', 'btw', 'mengambek', 'of', 'favoritmu',
    'janjinya', 'favoritmu', 'temanteman', 'hai', 'xiaomilovers', 'mencoba', 'ive', 'kak', 'mur',
    'dips', 'tanyarl', 'terjatuh', 'vs', 'notis', 'pap', 'indak', 'the', 'jir', 'kaleng', 'mah', 'kat',
    'lal', 'coy', 'wis', 'babu', 'dikasih', 'segininya', 'cok', 'fan', 'was', 'weh', 'rek', 'bjir', 'itb', 'pakai'
    ]

    vectorizer_model = CountVectorizer(stop_words=custom_stopwords)
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        language="multilanguage",
        top_n_words=8
    )
    topics, probs = topic_model.fit_transform(texts)
    return topic_model, topics

# === PLOT PIE CHART ===
def plot_sentiment_pie(df):
    st.subheader("\U0001F4C8 Distribusi Sentimen (Pie Chart)")

    sentiment_colors = {
        "positif": '#66bb6a',
        "netral": '#b0bec5',
        "negatif": '#ef5350'
    }

    sentiment_counts = df["sentiment"].value_counts()
    labels = sentiment_counts.index
    values = sentiment_counts.values
    colors = [sentiment_colors[label] for label in labels]

    fig, ax = plt.subplots(figsize=(1.5, 1.5))  # smaller figure
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops=dict(color="black", fontsize=6)  # smaller font
    )
    ax.axis("equal")
    ax.set_title("Distribusi Sentimen", fontsize=8)  # smaller title
    st.pyplot(fig)


def plot_sentiment_bar(df):
    sentiment_colors = {
        "positif": '#66bb6a',
        "netral": '#b0bec5',
        "negatif": '#ef5350'
    }

    sentiment_counts = df["sentiment"].value_counts().reindex(["positif", "netral", "negatif"], fill_value=0).reset_index()
    sentiment_counts.columns = ["sentiment", "count"]
    sentiment_counts["color"] = sentiment_counts["sentiment"].map(sentiment_colors)

    fig = px.bar(
        sentiment_counts,
        x="sentiment",
        y="count",
        color="sentiment",
        color_discrete_map=sentiment_colors,
        text="count",
        title="Distribusi Sentimen",
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Sentimen",
        yaxis_title="Jumlah",
        showlegend=False,
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)


# === PLOT SENTIMEN PER TOPIK ===
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

def show_filtered_dataframe(df):

    # Filter Sentiment
    sentiments = df['sentiment'].unique().tolist()
    selected_sentiment = st.selectbox("Filter berdasarkan Sentimen:", ["Semua"] + sentiments)

    # Filter Cluster
    cluster_ids = sorted(df['topic_id'].unique())
    selected_cluster = st.selectbox("Filter berdasarkan Topik/Cluster ID:", ["Semua"] + cluster_ids)

    filtered_df = df.copy()

    if selected_sentiment != "Semua":
        filtered_df = filtered_df[filtered_df["sentiment"] == selected_sentiment]

    if selected_cluster != "Semua":
        filtered_df = filtered_df[filtered_df["topic_id"] == selected_cluster]

    st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

# === MAIN APP ===
def main():
    st.title("\U0001F4D1 Analisis Sentimen & Topik Berbasis IndoBERT + BERTopic")
    uploaded_file = st.file_uploader("Unggah file CSV (timestamp & text)", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

        if set(df.columns) >= {"timestamp", "text"}:
            st.success("✅ Format kolom valid.")
            norm_dict = load_normalization_dict()
            stopwords = load_stopwords()
            tokenizer, model, class_names = load_sentiment_model()

            st.info("🔧 Membersihkan teks...")
            df["clean_text"] = df["text"].apply(lambda x: clean_text(x, norm_dict, stopwords))
            st.success("✅ Pembersihan selesai")

            st.info("📌 Memfilter teks relevan...")
            relevansi_tokenizer, relevansi_model = load_relevansi_model()
            df["relevan"] = predict_relevansi(df["clean_text"].tolist(), relevansi_tokenizer, relevansi_model)
            # Keep only relevan rows
            df = df[df["relevan"] == 1].reset_index(drop=True)
            st.success(f"✅ Ditemukan {len(df)} teks relevan dari {len(df) + df['relevan'].value_counts().get(0, 0)} total baris data.")

            st.info("⏳ Memproses prediksi sentimen...")
            df["sentiment"] = predict_sentiment_batch(df["clean_text"].tolist(), tokenizer, model, class_names)
            st.success("✅ Prediksi sentimen selesai")

            st.info("⏳ Memproses pemodelan topik...")
            topic_model, topics = run_bertopic_model(df["clean_text"].tolist())
            df["topic_id"] = topics
            st.success("✅ Pemodelan topik selesai")

            ## VISUALISASI DAN OUTPUT
            
            # Visualisasi Pie Chart Sentimen
            # st.subheader("📊 Distribusi Sentimen (Pie Chart)")
            # plot_sentiment_pie(df)

            # Visualisasi Bar Chart Sentimen
            st.subheader("📊 Distribusi Sentimen (Bar Chart Interaktif)")
            plot_sentiment_bar(df)

            # === Tampilkan Sampel 5 Teks untuk Setiap Sentimen ===
            st.subheader("📝 Contoh Teks Hasil Prediksi Berdasarkan Sentimen")

            for sent in class_names:  # ['negatif', 'netral', 'positif']
                st.markdown(f"**Sentimen: {sent.capitalize()}**")
                sample_df = df[df["sentiment"] == sent].sample(5)[["timestamp", "text"]]
                if not sample_df.empty:
                    st.dataframe(sample_df.reset_index(drop=True))
                else:
                    st.write("_Tidak ada data untuk sentimen ini._")


            # Visualisasi Distribusi Sentimen per Topik
            plot_sentiment_per_topic(df, topic_model)

            # Tampilkan tabel topik
            st.subheader("📋 Ringkasan Topik")
            # Extract clean keywords from topic_model
            def extract_keywords(topic_model, n=8):
                topics_dict = topic_model.get_topics()
                keyword_map = {}
                for topic_id, words in topics_dict.items():
                    keyword_list = [word for word, _ in words[:n]]
                    keyword_map[topic_id] = ", ".join(keyword_list)
                return keyword_map

            # Get cleaned topic info
            topic_info = topic_model.get_topic_info()
            keyword_map = extract_keywords(topic_model)

            # Add a 'keywords' column from the cleaned map
            topic_info["keywords"] = topic_info["Topic"].map(keyword_map)

            # Select and rename relevant columns
            topic_info = topic_info[["Topic", "keywords", "Count"]].rename(columns={
                "Topic": "cluster_id",
                "Count": "docs_count"
            })

            # Display in Streamlit
            st.dataframe(topic_info)

            # Tampilkan tabel topik
            st.subheader("📋 Dataset dengan Label")
            show_filtered_dataframe(df[['timestamp', 'text', 'sentiment', 'topic_id'])
        else:
            st.error("❌ File harus memiliki kolom: timestamp dan text")

if __name__ == "__main__":
    main()



