import matplotlib.pyplot as plt
import os
import streamlit as st
import pandas as pd
import re
import plotly.express as px  # For interactive pie charts

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
        .reset_index(name="Ticket Count")
    )
    return topic_counts

def build_category_summary(df):
    if "Category" in df.columns:
        return category_wise_count(df)
    else:
        return pd.DataFrame()

def display_summary_with_chart(summary_df, full_df, label_column, count_column):
    """Display table with percentage + interactive pie chart below."""
    if summary_df.empty:
        st.warning(f"No data available for {label_column}")
        return

    total_tickets = full_df["Ticket No."].nunique() if "Ticket No." in full_df.columns else 0
    if total_tickets > 0:
        summary_df["Percentage (%)"] = (summary_df[count_column] / total_tickets * 100).round(2)

    # Show table
    st.dataframe(summary_df.reset_index(drop=True), use_container_width=True, hide_index=True)

    # Plotly interactive pie chart with outside labels & tilt
    fig = px.pie(
        summary_df,
        names=label_column,
        values=count_column,
        hole=0.3,
        title=f"{label_column} Distribution",
    )
    fig.update_traces(
        textinfo="label+percent",
        textposition="outside",
        automargin=True,
        textfont=dict(size=12),
        pull=[0.05 if val < total_tickets * 0.05 else 0 for val in summary_df[count_column]],  # pull small slices
    )
    fig.update_layout(
        legend_title_text=label_column,
        height=500,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    st.plotly_chart(fig, use_container_width=True)


def display_clickable_summary(summary_df, full_df, key_prefix, label_column, count_column):
    display_summary_with_chart(summary_df, full_df, label_column, count_column)

    options = summary_df[label_column].tolist()
    selected = st.selectbox(f"Select {label_column}", options, key=f"{key_prefix}_select")
    if selected:
        st.markdown(f"### 🔍 Records for {label_column}: `{selected}`")
        filtered_df = full_df[full_df[label_column] == selected]
        st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True, hide_index=True)

def sort_agent_names(df, agent_column="Analyst who closed the ticket"):
    df = df.copy()
    df["Agent_num"] = df[agent_column].apply(
        lambda x: int(re.search(r'(\d+)$', str(x)).group(1)) if re.search(r'(\d+)$', str(x)) else -1
    )
    df = df.sort_values("Agent_num").drop(columns="Agent_num")
    return df

if selected_option == "Upload and Process New File":
    st.info("⚠️ Processing may take up to **30 minutes**. Please wait after uploading your file.")

    uploaded_file = st.file_uploader("Upload Excel File (PoorRatings-Tracker.xlsx)", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file, sheet_name="Sheet1")
        df.columns = df.columns.str.strip().str.replace('\n', ' ').str.replace('\r', ' ').str.title()

        # 1. Category
        st.subheader("1. Category")
        cat_df = build_category_summary(df)
        if not cat_df.empty and "Category" in cat_df.columns and "Ticket Count" in cat_df.columns:
            display_clickable_summary(cat_df, df, "category", "Category", "Ticket Count")
        else:
            st.warning("No category data available.")

        # 2. Sentiments
        st.subheader("2. Sentiments")
        sentiment_counts = build_sentiment_summary(df, recompute=True)
        display_clickable_summary(sentiment_counts, df, "sentiment", "Sentiment", "Ticket Count")

        # 3. Topic Labels
        st.subheader("3. Topic Labels using BERTopic + LLM")
        df = generate_topic_labels(df)
        topic_counts = build_topic_summary(df)
        display_clickable_summary(topic_counts, df, "topic", "Topic Label", "Ticket Count")

        # 4. Analyst Summary
        st.subheader("4. Analyst who closed the ticket")
        analyst_df = generate_analyst_summary(df)
        analyst_df = sort_agent_names(analyst_df)
        st.dataframe(analyst_df.reset_index(drop=True), hide_index=True)

        df.to_excel(output_path, index=False)
        st.success(f"✅ Output saved to `{output_path}`")

elif selected_option == "View Last Processed Output":
    if os.path.exists(output_path):
        df = pd.read_excel(output_path)

        # 1. Category
        st.subheader("1. Category")
        cat_df = build_category_summary(df)
        display_clickable_summary(cat_df, df, "category", "Category", "Ticket Count")

        # 2. Sentiments
        st.subheader("2. Sentiments")
        sentiment_counts = build_sentiment_summary(df, recompute=False)
        display_clickable_summary(sentiment_counts, df, "sentiment", "Sentiment", "Ticket Count")

        # 3. Topic Labels
        st.subheader("3. Topic Labels")
        topic_counts = build_topic_summary(df)
        display_clickable_summary(topic_counts, df, "topic", "Topic Label", "Ticket Count")

        # 4. Analyst Summary
        st.subheader("4. Analyst who closed the ticket")
        analyst_df = generate_analyst_summary(df)
        analyst_df = sort_agent_names(analyst_df)
        st.dataframe(analyst_df.reset_index(drop=True), hide_index=True)

    else:
        st.warning("⚠️ No previously processed file found.")

else:
    st.info("👇 Please select an option to begin.")
