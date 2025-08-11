import pandas as pd
from bertopic import BERTopic
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
import re
from tqdm import tqdm
import streamlit as st

def clean_label(label):
    return re.sub(r"^\d+_", "", label).replace("_", " ")

@st.cache_resource(show_spinner=False)
def load_bertopic():
    # Initialize BERTopic once
    return BERTopic(language="english", calculate_probabilities=False, verbose=False)

@st.cache_data(show_spinner=False)
def run_bertopic(texts):
    topic_model = load_bertopic()
    topics, probs = topic_model.fit_transform(texts)
    new_labels = topic_model.generate_topic_labels(nr_words=3, topic_prefix=False, separator=" | ")
    topic_model.set_topic_labels(new_labels)
    return topics, topic_model

@st.cache_data(show_spinner=False)
def refine_labels_with_llm(unique_labels_df):
    prompt_template = PromptTemplate.from_template(
        "You are a helpful assistant. Rewrite the following topic label to make it human-readable "
        "and concise (max 5 words).\n\n"
        "Examples:\n"
        "ticket_the_was_closed → Closed Ticket\n"
        "software_the_to_it → Software Upgrade Process\n"
        "access_to_this_permissions → Permission Access\n"
        "password_to_the_phone → Phone Password\n"
        "problem_not_resolved_issue → Unresolved Issue\n\n"
        "Raw: {label}\nImproved:"
    )
    llm = Ollama(model="mistral")
    final_labels = {}
    for _, row in tqdm(unique_labels_df.iterrows(), total=len(unique_labels_df)):
        topic_id = row["Topic ID"]
        cleaned = clean_label(row["Topic Label"])
        prompt = prompt_template.format(label=cleaned)
        improved = llm.invoke(prompt).strip().replace("\n", " ")
        final_labels[topic_id] = improved
    return final_labels

def generate_topic_labels(df):
    df = df[df["User Response"].notna()]
    df = df[~df["User Response"].str.strip().str.lower().isin(["", "no comments from the user"])]
    texts = df["User Response"].tolist()

    topics, topic_model = run_bertopic(texts)
    df["Topic ID"] = topics
    df["Topic Label"] = [topic_model.topic_labels_.get(t, "Unknown") for t in topics]
    df = df[df["Topic ID"] != -1]

    unique_labels = df[["Topic ID", "Topic Label"]].drop_duplicates()
    final_labels = refine_labels_with_llm(unique_labels)
    df["Topic Label"] = df["Topic ID"].map(final_labels)

    return df
