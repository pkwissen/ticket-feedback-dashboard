import os
import streamlit as st
import pandas as pd
from modules.sentiment import classify_sentiment, labels as sentiment_labels
from modules.topic_modeling import generate_topic_labels
from modules.analyst_summary import generate_analyst_summary
from modules.category_count import category_wise_count

# Page config
st.set_page_config(page_title="Ticket Feedback Dashboard", layout="wide")
st.title("🎫 Ticket Feedback Dashboard (Poor Ratings)")

# Paths
project_base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(project_base_dir, "processed_output")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "feedback_with_topics_and_sentiment.xlsx")

selected_option = st.radio(
    "Choose an option",
    ["-- Select an option --", "Upload and Process New File", "View Last Processed Output"],
    index=0
)


def build_sentiment_summary(df, recompute=False):
    if recompute or "Sentiment" not in df.columns:
        df["Sentiment"] = df["User Response"].fillna("").astype(str).apply(classify_sentiment)
    else:
        df["Sentiment"] = df["Sentiment"].fillna("Unclear")
    sentiment_counts = (
        df["Sentiment"]
        .value_counts()
        .reindex(sentiment_labels, fill_value=0)
        .rename_axis("Sentiment")
        .reset_index(name="Ticket Count")
    )
    return sentiment_counts


def build_topic_summary(df):
    if "Topic Label" not in df.columns:
        return pd.DataFrame()
    topic_counts = (
        df["Topic Label"]
        .value_counts()
        .rename_axis("Topic Label")
        .reset_index(name="count")
    )
    return topic_counts


def build_category_summary(df):
    if "Category" in df.columns:
        return category_wise_count(df)
    else:
        return pd.DataFrame()

if selected_option == "Upload and Process New File":
    st.info("⚠️ Processing may take up to **30 minutes**. Please wait after uploading your file.")
    
    uploaded_file = st.file_uploader("Upload Excel File (PoorRatings-Tracker.xlsx)", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file, sheet_name="Sheet1")


        # 1. Category
        st.subheader("1. Category")
        st.caption("📌 Classification of user feedback into predefined issue types for analysis.")
        cat_df = build_category_summary(df)
        st.dataframe(cat_df.reset_index(drop=True), hide_index=True)

        # 2. Sentiments
        st.subheader("2. Sentiments")
        st.caption("📌 Classification of user feedback based on expressed attitude.")
        sentiment_counts = build_sentiment_summary(df, recompute=True)
        st.dataframe(sentiment_counts, hide_index=True)

        # 3. Topic Labels using BERTopic + LLM
        st.subheader("3. Topic Labels using BERTopic + LLM")
        st.caption("📌 Categorization of user feedback into key themes or topics (generated through topic modelling).")
        df = generate_topic_labels(df)
        topic_counts = build_topic_summary(df)
        st.dataframe(topic_counts, hide_index=True)

        # 4. Analyst who closed the ticket
        st.subheader("4. Analyst who closed the ticket")
        st.caption("📌 Monthly ticket statistics for each analyst who closed a ticket.")
        analyst_df = generate_analyst_summary(df)
        st.dataframe(analyst_df.reset_index(drop=True), hide_index=True)

        df.to_excel(output_path, index=False)
        st.success(f"✅ Output saved to `{output_path}`")

elif selected_option == "View Last Processed Output":
    if os.path.exists(output_path):
        df = pd.read_excel(output_path)
        st.success("📂 Loaded last processed output")

        # 1. Category
        st.subheader("1. Category")
        st.caption("📌 Classification of user feedback into predefined issue types for analysis.")
        cat_df = build_category_summary(df)
        st.dataframe(cat_df.reset_index(drop=True), hide_index=True)

        # 2. Sentiments
        st.subheader("2. Sentiments")
        st.caption("📌 Classification of user feedback based on expressed attitude.")
        if "Sentiment" not in df.columns:
            st.warning("Sentiment column missing in saved file; computing now.")
        sentiment_counts = build_sentiment_summary(df, recompute=False)
        st.dataframe(sentiment_counts, hide_index=True)

        # 3. Topic Labels
        st.subheader("3. Topic Labels")
        st.caption("📌 Categorization of user feedback into key themes or topics (generated through topic modelling).")
        topic_counts = build_topic_summary(df)
        st.dataframe(topic_counts, hide_index=True)

        # 4. Analyst who closed the ticket
        st.subheader("4. Analyst who closed the ticket")
        st.caption("📌 Monthly ticket statistics for each analyst who closed a ticket.")
        analyst_df = generate_analyst_summary(df)
        st.dataframe(analyst_df.reset_index(drop=True), hide_index=True)
    else:
        st.warning("⚠️ No previously processed file found.")
else:
    st.info("👆 Please select an option to begin.")
