import os
import re
import socket
import pandas as pd
from tqdm import tqdm
from bertopic import BERTopic
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq


def is_ollama_running(host="127.0.0.1", port=11434, timeout=0.5):
    """Check if Ollama is running locally."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def clean_label(label):
    """Remove underscores, extra spaces, and lowercase."""
    return re.sub(r'\s+', ' ', label.replace("_", " ")).strip().lower()


def refine_labels_with_llm(unique_labels_df):
    """
    Refine BERTopic labels using either Ollama locally or Groq on Streamlit Cloud.
    """
    prompt_template = PromptTemplate.from_template(
        "You are a helpful assistant. Rewrite the following topic label to make it "
        "human-readable and concise (max 5 words).\n\n"
        "Examples:\n"
        "ticket_the_was_closed → Closed Ticket\n"
        "software_the_to_it → Software Upgrade Process\n"
        "access_to_this_permissions → Permission Access\n"
        "password_to_the_phone → Phone Password\n"
        "problem_not_resolved_issue → Unresolved Issue\n\n"
        "Raw: {label}\nImproved:"
    )

    # Choose LLM backend
    if is_ollama_running():
        # print("✅ Using local Ollama (mistral)")
        llm = Ollama(model="mistral", temperature=0)
    else:
        # print("☁️ Using Groq (mixtral-8x7b-32768)")
        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        llm = ChatGroq(
            model="mixtral-8x7b-32768",
            api_key=groq_key,
            temperature=0
        )

    final_labels = {}
    for _, row in tqdm(unique_labels_df.iterrows(), total=len(unique_labels_df)):
        topic_id = row["Topic ID"]
        cleaned = clean_label(row["Topic Label"])
        prompt = prompt_template.format(label=cleaned)
        try:
            improved = llm.invoke(prompt).strip().replace("\n", " ")
        except Exception as e:
            print(f"Error refining label for topic {topic_id}: {e}")
            improved = cleaned
        final_labels[topic_id] = improved

    return final_labels


def generate_topic_labels(df, text_column="Feedback"):
    """
    Generate BERTopic topics and refined human-readable labels.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")

    # Fit BERTopic
    topic_model = BERTopic(language="english", calculate_probabilities=False, verbose=True)
    topics, _ = topic_model.fit_transform(df[text_column])

    # Add topic IDs to df
    df["Topic ID"] = topics
    df = df[df["Topic ID"] != -1]  # drop outliers

    # Get unique topic labels
    unique_labels = pd.DataFrame({
        "Topic ID": df["Topic ID"].unique(),
        "Topic Label": [topic_model.get_topic(topic)[0][0] for topic in df["Topic ID"].unique()]
    })

    # Refine labels with LLM
    final_labels = refine_labels_with_llm(unique_labels)

    # Map back to DataFrame
    df["Refined Topic Label"] = df["Topic ID"].map(final_labels)

    return df
